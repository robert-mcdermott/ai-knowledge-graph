# 🚀 部署指南

本指南說明如何將 AI Knowledge Graph Generator 部署到雲端平台。

## 📋 部署前準備

1. **GitHub 代碼庫**：確保代碼已推送到 GitHub
2. **API Key**：準備好您的 LLM API key（Groq/OpenAI/Anthropic）

---

## ☁️ 部署選項

### **選項 1: Railway（推薦）**

#### 為什麼選擇 Railway？
- ✅ 免費額度充足（每月 $5 credit）
- ✅ 自動從 GitHub 部署
- ✅ 簡單易用的界面
- ✅ 支援環境變數

#### 部署步驟

1. **註冊 Railway**
   - 訪問 https://railway.app
   - 使用 GitHub 帳號登入

2. **創建新項目**
   - 點擊 "New Project"
   - 選擇 "Deploy from GitHub repo"
   - 選擇您的 `ai-knowledge-graph` 代碼庫

3. **設置環境變數**
   點擊項目 → Variables → 添加以下環境變數：
   ```
   LLM_MODEL=llama-3.1-70b-versatile
   LLM_API_KEY=gsk_your_groq_api_key_here
   LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   ```

4. **部署**
   - Railway 會自動檢測 `railway.json` 並開始部署
   - 等待幾分鐘讓部署完成
   - 獲取公開 URL（例如：`your-app.railway.app`）

5. **訪問您的應用**
   - 打開 Railway 提供的 URL
   - 開始上傳文件生成知識圖譜！

---

### **選項 2: Render**

#### 為什麼選擇 Render？
- ✅ 完全免費（但有限制）
- ✅ 自動 HTTPS
- ✅ 從 GitHub 自動部署

#### 部署步驟

1. **註冊 Render**
   - 訪問 https://render.com
   - 使用 GitHub 帳號登入

2. **創建新 Web Service**
   - 點擊 "New" → "Web Service"
   - 連接您的 GitHub 代碼庫
   - 選擇 `ai-knowledge-graph` 代碼庫

3. **配置服務**
   ```
   Name: ai-knowledge-graph
   Region: Oregon (或您偏好的地區)
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.knowledge_graph.api.app:app --host 0.0.0.0 --port $PORT
   ```

4. **設置環境變數**
   在 Environment 選項卡添加：
   ```
   LLM_MODEL=llama-3.1-70b-versatile
   LLM_API_KEY=gsk_your_groq_api_key_here
   LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   ```

5. **選擇免費方案**
   - 選擇 "Free" plan
   - 點擊 "Create Web Service"

6. **部署完成**
   - 等待部署（約 5-10 分鐘）
   - Render 會提供一個 `.onrender.com` URL

⚠️ **注意**：免費方案會在無活動時自動休眠，首次訪問可能需要等待 30 秒喚醒。

---

### **選項 3: Fly.io**

#### 為什麼選擇 Fly.io？
- ✅ 免費額度（3 個小型應用）
- ✅ 全球 CDN
- ✅ 支援 Docker

#### 部署步驟

1. **安裝 Fly CLI**
   ```bash
   # macOS/Linux
   curl -L https://fly.io/install.sh | sh

   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```

2. **登入 Fly**
   ```bash
   fly auth login
   ```

3. **初始化應用**
   ```bash
   cd /path/to/ai-knowledge-graph
   fly launch --no-deploy
   ```

4. **設置環境變數**
   ```bash
   fly secrets set LLM_MODEL=llama-3.1-70b-versatile
   fly secrets set LLM_API_KEY=gsk_your_groq_api_key_here
   fly secrets set LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
   ```

5. **部署**
   ```bash
   fly deploy
   ```

6. **獲取 URL**
   ```bash
   fly status
   ```

---

## 🔐 環境變數說明

### 必需的環境變數

| 變數名稱 | 說明 | 範例 |
|---------|------|------|
| `LLM_MODEL` | 使用的 LLM 模型 | `llama-3.1-70b-versatile` |
| `LLM_API_KEY` | API 金鑰 | `gsk_xxx...` (Groq) 或 `sk-xxx...` (OpenAI) |
| `LLM_BASE_URL` | API 端點 URL | `https://api.groq.com/openai/v1/chat/completions` |

### 可選的環境變數

| 變數名稱 | 說明 | 預設值 |
|---------|------|--------|
| `LLM_MAX_TOKENS` | 最大 token 數 | `8192` |
| `LLM_TEMPERATURE` | 生成溫度 | `0.8` |
| `PORT` | 服務監聽端口 | `8000` |

---

## 🎯 不同 LLM 服務的配置

### Groq（推薦，免費）
```bash
LLM_MODEL=llama-3.1-70b-versatile
LLM_API_KEY=gsk_xxxxxxxxxxxxx
LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
```

### OpenAI
```bash
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxxxxxxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

### Anthropic Claude
```bash
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_API_KEY=sk-ant-xxxxxxxxxxxxx
LLM_BASE_URL=https://api.anthropic.com/v1/messages
```

---

## 🐛 故障排除

### 應用無法啟動
- 檢查環境變數是否正確設置
- 查看部署日誌尋找錯誤訊息
- 確認 requirements.txt 包含所有依賴

### API 調用失敗
- 驗證 API key 是否有效
- 檢查 API 配額是否用完
- 確認 base_url 正確

### 記憶體不足
- 減少 chunk_size（在 Web 界面設置）
- 升級到付費方案以獲得更多資源

---

## 📊 成本估算

### Railway
- 免費額度：每月 $5 credit
- 預估用量：小說分析約 0.01-0.05 credit/次
- **月處理量**：約 100-500 次分析（免費）

### Render
- 完全免費
- 限制：自動休眠、較慢的啟動時間

### Fly.io
- 免費額度：3 個小型應用
- 預估用量：輕量使用完全免費

### LLM API 成本（Groq）
- **完全免費**
- 免費額度：每天 14,400 請求
- 適合個人和小規模使用

---

## 🔄 更新部署

### Railway / Render
- 推送代碼到 GitHub 主分支會自動觸發重新部署
- 或在平台界面手動觸發部署

### Fly.io
```bash
fly deploy
```

---

## 🌍 自定義域名（可選）

所有平台都支援自定義域名：

1. **購買域名**（如 Namecheap, GoDaddy）
2. **在部署平台設置自定義域名**
3. **更新 DNS 記錄**指向平台提供的地址

---

## 📚 更多資訊

- [Railway 文檔](https://docs.railway.app/)
- [Render 文檔](https://render.com/docs)
- [Fly.io 文檔](https://fly.io/docs/)
- [FastAPI 部署指南](https://fastapi.tiangolo.com/deployment/)

---

## 💡 建議

1. **開發環境**：先在本地測試（localhost:8000）
2. **測試環境**：部署到免費平台測試
3. **生產環境**：確認穩定後，考慮升級到付費方案以獲得更好性能

---

## 🆘 需要幫助？

如果遇到問題：
1. 查看部署平台的日誌
2. 檢查 [GitHub Issues](https://github.com/yourusername/ai-knowledge-graph/issues)
3. 確認所有環境變數正確設置

祝您部署順利！🚀
