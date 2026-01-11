# GitHub上传指南

## 步骤1：在GitHub上创建新仓库

1. 登录GitHub账号
2. 点击右上角的 **"+"** 按钮，选择 **"New repository"**
3. 填写仓库信息：
   - **Repository name**: 例如 `3d-polycrystal-simulation` 或 `crystal-microstructure`
   - **Description**: 例如 "GPU加速的3D多晶体微结构生成与模拟"
   - **Visibility**: 选择 Public（公开）或 Private（私有）
   - **不要**勾选 "Initialize this repository with a README"（我们已经有了）
4. 点击 **"Create repository"**

## 步骤2：连接本地仓库到GitHub

在创建仓库后，GitHub会显示一个页面，上面有仓库的URL。复制这个URL（例如：`https://github.com/你的用户名/仓库名.git`）

然后在本地项目目录执行以下命令：

```bash
# 添加远程仓库（将 YOUR_USERNAME 和 REPO_NAME 替换为你的实际信息）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用SSH（如果你配置了SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 查看远程仓库配置
git remote -v
```

## 步骤3：推送到GitHub

```bash
# 推送代码到GitHub（首次推送）
git push -u origin master

# 如果GitHub默认分支是main，使用：
# git branch -M main
# git push -u origin main
```

## 步骤4：验证上传

1. 刷新GitHub仓库页面
2. 你应该能看到所有文件已经上传
3. README.md会自动显示在仓库首页

## 后续更新代码

如果以后修改了代码，使用以下命令更新GitHub：

```bash
# 查看修改的文件
git status

# 添加所有修改的文件
git add .

# 提交修改（使用有意义的提交信息）
git commit -m "描述你的修改内容"

# 推送到GitHub
git push
```

## 常见问题

### 1. 如果提示需要身份验证

GitHub现在要求使用Personal Access Token（个人访问令牌）而不是密码：

1. 进入 GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token"
3. 选择权限（至少需要 `repo` 权限）
4. 生成后复制token
5. 在推送时，用户名输入你的GitHub用户名，密码输入刚才生成的token

### 2. 如果遇到分支名称问题

如果GitHub默认分支是 `main` 而本地是 `master`：

```bash
# 重命名本地分支
git branch -M main

# 推送并设置上游
git push -u origin main
```

### 3. 如果路径中有中文字符导致问题

如果遇到路径编码问题，可以：
- 在Git Bash中执行命令（而不是PowerShell）
- 或者使用GitHub Desktop等图形化工具

## 使用GitHub Desktop（可选）

如果你更喜欢图形界面：

1. 下载并安装 [GitHub Desktop](https://desktop.github.com/)
2. 登录你的GitHub账号
3. 点击 "File" → "Add local repository"
4. 选择你的项目文件夹
5. 点击 "Publish repository" 按钮上传到GitHub

