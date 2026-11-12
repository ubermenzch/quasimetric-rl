# ICLR 2026 论文阅读包

`REPORT_ZH.md` 包含 10 篇与 NQRL 相关论文的逐篇叙事与语言分析，以及对当前稿件的写作建议。

## 文件

- `REPORT_ZH.md`：中文分析报告，含原文链接和对应章节。
- `papers.json`：10 篇主会论文的官方条目、PDF 链接、阅读版本。
- `download_papers.py`：Python 3.8+ 标准库下载器，无需安装第三方依赖。
- `download_status.json`：下载运行结果；该文件记录实际成功数量。
- `pdfs/`：下载成功后存放原始 PDF。

## 下载 PDF

当前服务器终端的网络限制阻止了原始 PDF 下载。本次已通过联网阅读工具分析正文，初始 ZIP 包含报告、清单、脚本与状态记录，**不包含原始论文 PDF**。

在可以访问 ICLR proceedings 的电脑上解压，然后在此目录运行：

```bash
python3 download_papers.py --zip iclr2026_nqrl_with_pdfs.zip
```

Windows 可使用 `py -3` 替换 `python3`。下载完成后应显示 `Complete PDFs available: 10/10`。脚本检查 PDF 文件头和结束标记，记录大小及 SHA-256，重跑会跳过完整文件。失败时返回非零退出码；即使生成了 ZIP，也应检查状态记录，不能仅凭存在 ZIP 判断 10 篇均已下载。

PDF 来自官方会议归档。由于部分官方 PDF 无法通过阅读工具加载，本次对应分析阅读了作者的 arXiv HTML 版本；具体版本见清单。正文分析完成与原始 PDF 落盘状态分别记录。

## 从服务器下载本阅读包

在自己的电脑终端执行：

```bash
scp -P 8899 zhangcheng@10.82.1.226:/home/zhangcheng/quasimetric-rl/iclr2026_nqrl_reading_pack.zip .
```

这是研究资料包，不是 LoongLaTeX 投稿包；不会替换 `nqrl_iclr2027_loonglatex.zip`。
