---
title: n8n Workflow Generator
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 n8n Workflow Generator

AI-powered n8n workflow generation using **Llama 3 8B** fine-tuned on **6,000+ real workflows** from the n8n marketplace!

## 🎯 What This Does

Describe what you want to automate in plain English, and the AI generates a complete n8n workflow configuration for you!

**Examples:**
- "Build a Telegram chatbot that uses OpenAI to respond to messages"
- "Create a workflow that monitors Gmail and sends Slack notifications"
- "Build an automation that scrapes prices and saves to Google Sheets"

## 📚 The Story Behind This Model

This model was trained through an extensive data science project:

1. **Part 1**: [Scraped 6,000+ n8n workflows](https://medium.com/@mustaphaliaichi/what-are-people-actually-building-in-n8n-i-scraped-over-6-000-workflows-to-find-out-59eb8e34c317) from the marketplace
2. **Part 2**: [Fine-tuned Llama 3 8B](https://medium.com/@mustaphaliaichi/i-fine-tuned-llama-3-on-6-000-n8n-workflows-after-mistral-failed-spectacularly-927cce57df14) after Mistral failed spectacularly (loss: 1.235900)
3. **Part 3**: This deployment! 🎉

## 🚀 How to Use

1. Click **"Load Model"** (takes ~30 seconds first time)
2. Describe your desired automation
3. Click **"Generate Workflow"**
4. Copy the JSON and import into n8n!

## 📊 Model Details

- **Base Model**: Llama 3 8B
- **Training Data**: 6,000+ real n8n workflows
- **Fine-tuning Method**: LoRA with Unsloth
- **Final Training Loss**: 1.235900
- **Training Time**: ~55 minutes on A100

## 🔗 Resources

- 🤗 **Model**: [MustaphaL/n8n-workflow-generator](https://huggingface.co/MustaphaL/n8n-workflow-generator)
- 🤗 **Dataset**: [MustaphaL/n8n-workflow-training-data](https://huggingface.co/datasets/MustaphaL/n8n-workflow-training-data)
- 📚 **Learn n8n**: [n8nlearninghub.com](https://n8nlearninghub.com)
- 💬 **Community**: [r/n8nLearningHub](https://reddit.com/r/n8nLearningHub) (1,000+ members)

## ⚠️ Important Notes

- This is AI-generated content - always review before production use
- Generated workflows may need credential configuration
- Test thoroughly before deploying
- Model may occasionally need manual adjustments

## 👨‍💻 Created By

**Mustapha LIAICHI** (MHL)
- AI Engineer at Sweet Spot Consulting Ltd
- Founder of n8n Learning Hub
- Reddit community: 1,000+ members
- Weekly n8n tutorials published

## 📝 License

MIT License - Feel free to use and modify!

---

**Built with ❤️ using Llama 3 8B + Unsloth + Gradio**
