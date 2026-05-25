---
name: git-push
description: 帮助用户完成完整的 git 提交工作流（add + commit + push）。会基于历史 commit 风格和当前改动自动生成中文 commit message。用户说「提交」「push 一下」「帮我 commit」「git push」「把改动推上去」等场景触发。
---

# git-push 工作流 Skill

你的任务是帮助用户走完 **add → commit → push** 三步，每一步都需要用户显式确认后才能执行。

## 核心原则

1. **commit message 用中文**，匹配本仓库历史风格
2. **三个阶段（add / commit / push）都要让用户确认**，不要一口气全做完
3. **绝对禁止** `--no-verify`、`--force`、`--amend` 已推送的 commit、`git add .` / `git add -A`（容易把 .env 之类带进去）
4. **不修改 git config**

## 执行步骤

### Step 1 — 摸清当前状态（并行执行）

并行运行下面四个命令，把仓库状态搞清楚：

```bash
git status                     # 看哪些文件改了 / 新增 / 删除
git diff                       # 未暂存的改动
git diff --staged              # 已暂存的改动
git log --oneline -8           # 学习历史 commit 的语言和风格
```

如果当前已经在某个 detached HEAD 或者很奇怪的分支状态，先提示用户、停下来。

### Step 2 — 分析改动并提议暂存范围

读完 diff 和 status 之后，你需要：

1. **按"逻辑单元"理解改动**：不是按文件，是按"这一坨改动想干啥"。比如同时改了 `kernel.cu` 和 `test_cfg.py` 通常是一个事。
2. **挑出可能不该提交的东西**：
   - `.env`、`*.key`、`credentials*`、`secrets*` 等敏感文件 → 必须警告并默认排除
   - `build/`、`*.o`、`*.so`、`__pycache__/`、`reports/`、`plots/`、`*.nsys-rep`、`*.ncu-rep` 等构建产物 → 默认排除并提示
   - 大文件（>5MB 的非源码文件） → 提示用户确认
3. **如果发现存在多个不相关的改动**，建议用户分多次 commit；不要硬塞成一个。

向用户展示：
- 准备 `git add` 的文件清单
- 排除的文件以及排除原因
- 如果有可疑文件，**明确列出来让用户拍板**

得到用户确认后再执行 `git add <具体文件路径>`（**永远列具体路径，禁止用 `.` 或 `-A`**）。

### Step 3 — 生成 commit message

**先看历史风格**（已经在 Step 1 拿到了 `git log`），常见样本：

```
添加了环境配置文件，修复了 device 查询相关的问题
update matmul code
优化了下逻辑，不再盲目做笛卡尔积，重点放在算子实现上
完善了 07-rmsnorm 的相关实现，优化了torch compile的问题
feat: V2 架构重构
```

风格特征：**中文为主、口语化、说清楚做了什么+为什么，偶尔用 `feat:` / `fix:` 前缀**。生成新 message 时遵循这个风格。

**生成规则**：

- **一句话主旨**（必须）：动词开头，说清楚这次提交干了啥。例如「优化了 matmul 的 swizzle 实现」「修复了 device 查询在多卡环境下的崩溃」
- **如果改动超过一个逻辑单元**：主旨之后空一行，用 `-` 列点说明每个子项
- **避免空话**：不要写「一些修改」「部分更新」这种没有信息量的内容。如果实在没法总结清楚，说明改动太杂，应该建议用户拆分 commit
- **聚焦"为什么"而非"什么"**：代码本身能看出"改了什么"，commit message 要补充"为什么这么改"。例如对比：
  - ❌「修改了 strategies.py 的 _resolve_metrics 函数」
  - ✅「修复 metrics=all 时没把 bw/flops 别名展开导致绘图缺失的问题」

**展示给用户的格式**：

```
📝 拟定的 commit message：

    <生成的 message>

确认提交？(yes / 修改 / 取消)
```

用户可能回答：
- `yes` / `好` / `提交吧` → 执行 commit
- `改成 XXX` / `换成 ...` → 用户给新的 message，再确认一次
- `取消` / `算了` → 停下，不 commit

确认后执行：

```bash
git commit -m "$(cat <<'EOF'
<commit message>
EOF
)"
```

**注意**：不要附加 `Co-Authored-By: Claude` 之类的尾签——这是用户的个人仓库，按他的历史风格不带这些。

如果 commit 因为 pre-commit hook 失败：**先修问题，再创建新 commit**，绝不用 `--no-verify`，也绝不 `--amend`（pre-commit 失败意味着 commit 没产生，amend 会改到上一个无关 commit）。

### Step 4 — push 前再确认一次

commit 完成后：

1. 跑 `git status` 和 `git log -1` 确认 commit 落地
2. 检查当前分支和远程：`git rev-parse --abbrev-ref HEAD` + `git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null`
3. 如果是 `main` / `master` 分支，**额外强提示一次**：「即将 push 到主分支，确认？」
4. 如果上游分支不存在，命令要带 `-u`：`git push -u origin <branch>`
5. **绝对禁止** `git push --force` / `--force-with-lease`，除非用户明确指名要强推（并再次确认目标不是主分支）

给用户展示：

```
🚀 准备 push：
    本地：<current-branch>  (新 commit: <hash> <subject>)
    远程：<remote>/<branch>
    命令：git push [-u origin <branch>]

确认 push？(yes / 取消)
```

确认后执行 push，把输出原样回显给用户。

## 异常处理

- **没有任何改动**：提示用户「工作区干净，没有需要提交的内容」，结束
- **只有未跟踪的文件**：照常列出来让用户决定是否 add
- **存在合并冲突标记 (`<<<<<<<`)**：拒绝继续，提示用户先解决冲突
- **远程有新 commit（push 会被拒）**：建议用户先 `git pull --rebase`，**不要**替用户执行这个操作（涉及 rebase，让用户自己决策）

## 简洁度要求

- 每一步只输出该步的关键信息，不要重复 git 输出
- 不要在每个步骤都列一遍"接下来我要做..."这种元叙述
- 用户确认完成一步后，直接进入下一步，不需要"好的"「明白了」之类的过渡话
