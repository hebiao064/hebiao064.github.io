# 中文版博客设置指南 / Chinese Blog Setup Guide

## 概述 / Overview

您的Jekyll博客现在支持双语内容！通过简单的配置，您可以选择性地添加中文博客文章。

Your Jekyll blog now supports bilingual content! With simple configuration, you can selectively add Chinese blog posts.

## 如何添加中文文章 / How to Add Chinese Posts

### 1. 在文章前言中添加语言标签 / Add Language Tag in Front Matter

在您想要设为中文的文章的YAML前言中添加 `lang: zh`：

Add `lang: zh` to the YAML front matter of posts you want to mark as Chinese:

```yaml
---
title: 您的中文标题
updated: 2025-01-15 12:00
lang: zh
---
```

### 2. 正常编写文章内容 / Write Article Content Normally

之后就可以正常用中文编写文章内容了。

Then write your article content in Chinese as normal.

## 功能特性 / Features

- **双语导航**: 页面顶部有"English | 中文版"切换选项
- **智能过滤**: 
  - 主页 (`/`) 显示所有英文文章（没有`lang: zh`标签的文章）
  - 中文版页面 (`/chinese/`) 只显示带有`lang: zh`标签的文章
- **统一设计**: 保持与原主题一致的设计风格
- **响应式**: 支持移动端显示

## 文件结构 / File Structure

```
├── chinese.html          # 中文版首页
├── _includes/
│   └── header.html       # 包含导航栏的头部文件
├── _sass/
│   └── main.scss         # 包含导航栏样式
└── _posts/
    ├── 2025-01-15-sample-chinese-post.md  # 示例中文文章
    └── ...               # 其他文章
```

## 示例 / Examples

查看 `_posts/2025-01-15-sample-chinese-post.md` 文件来了解如何创建中文文章。

Check the `_posts/2025-01-15-sample-chinese-post.md` file to see how to create Chinese articles.

## 使用建议 / Usage Tips

1. **保持一致性**: 建议为每篇中文文章都添加 `lang: zh` 标签
2. **标题命名**: 可以在文件名中使用拼音或英文，但标题可以是中文
3. **日期格式**: 保持与现有文章相同的日期命名格式

---

现在您可以开始添加中文博客文章了！如有问题，请检查示例文章的格式。
