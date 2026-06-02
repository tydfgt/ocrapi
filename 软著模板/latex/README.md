# LaTeX 软著模板使用说明

本文件夹包含软件著作权申请文档的 **LaTeX** 格式模板，可通过 `xelatex` 编译生成高质量 PDF 文件。

## 目录结构

```
latex/
├── README.md
├── 01-申请表/
│   └── 01-申请表.tex          # 软件著作权登记申请表
├── 02-用户手册/
│   └── 02-用户手册.tex        # 软件说明书（用户手册）
├── 03-源代码说明/
│   └── 03-源代码说明.tex      # 源代码文档提交规范说明
├── 04-设计说明书/
│   └── 04-设计说明书.tex      # 软件设计说明书
└── 05-材料清单/
    └── 05-材料清单.tex        # 申请材料清单与自查表
```

## 环境要求

需要安装以下 LaTeX 发行版之一：

- **TeX Live**（推荐，跨平台）：`sudo apt install texlive-full`（Linux）
- **MiKTeX**（Windows）
- **MacTeX**（macOS）

### 必需宏包

| 宏包 | 用途 |
|------|------|
| `ctex` | 中文支持 |
| `geometry` | 页面边距设置 |
| `longtable` | 跨页表格 |
| `hyperref` | 超链接 |
| `listings` | 代码高亮（03 使用） |
| `tikz` | 架构图绘制（04 使用） |
| `graphicx` | 图片插入 |
| `fancyhdr` | 页眉页脚 |
| `caption` | 图表标题 |

> 安装 TeX Live Full 版本即可包含以上所有宏包。

## 编译方式

```bash
# 编译单个文件（需两次编译以生成目录）
cd 01-申请表
xelatex 01-申请表.tex
xelatex 01-申请表.tex

# 或使用 latexmk 自动编译
latexmk -xelatex 01-申请表.tex
```

## 填写说明

1. 用编辑器打开 `.tex` 文件
2. 搜索 `【` 占位符，替换为实际内容
3. 替换截图占位区域为实际图片：将 `\includegraphics{...}` 中的路径替换为你的截图文件
4. 编译生成 PDF

## 常见问题

**Q: 编译报错 "ctexart.cls not found"**
A: 未安装 ctex 宏包，执行 `sudo apt install texlive-lang-chinese` 或安装 texlive-full。

**Q: 中文不显示/乱码**
A: 确保使用 `xelatex` 而非 `pdflatex` 编译，且系统安装了中文字体。

**Q: 图片不显示**
A: 确保图片文件路径正确，支持 PNG/JPG/PDF 格式。
