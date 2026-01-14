# 🌐 Web 界面使用指南

這是 AI Knowledge Graph Generator 的 Web 界面，讓您可以通過瀏覽器輕鬆上傳文本並生成知識圖譜。

## ✨ 功能特點

- 📤 **拖拽上傳** - 支持拖拽或點擊選擇文件
- ⚙️ **自定義配置** - 可調整文本塊大小、重疊、標準化和推理選項
- 🔄 **實時進度** - 通過 WebSocket 實時顯示處理進度
- 📊 **統計展示** - 顯示節點數、關係數等統計信息
- 🎨 **互動視覺化** - 直接在瀏覽器中查看生成的知識圖譜
- ⬇️ **下載結果** - 下載 HTML 格式的知識圖譜

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

或使用 uv：

```bash
uv sync
```

### 2. 配置 LLM 設置

編輯 `config.toml` 文件，配置您的 LLM 服務：

```toml
[llm]
model = "gemma3"
api_key = "sk-1234"
base_url = "http://localhost:11434/v1/chat/completions"
max_tokens = 8192
temperature = 0.2
```

支持的 LLM 服務：
- Ollama (本地運行)
- LM Studio
- OpenAI API
- vLLM
- LiteLLM (支持 AWS Bedrock, Azure OpenAI, Anthropic 等)

### 3. 啟動 Web 服務器

```bash
python start_web.py
```

或者：

```bash
python -m uvicorn src.knowledge_graph.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 打開瀏覽器

訪問 http://localhost:8000

## 📖 使用說明

### 上傳文件

1. **選擇文件**
   - 點擊上傳區域選擇文件
   - 或直接拖拽文件到上傳區域
   - 支持 `.txt` 和 `.md` 格式

2. **配置處理參數**
   - **文本塊大小 (Chunk Size)**: 每個文本塊的字數 (預設: 200)
   - **重疊字數 (Overlap)**: 文本塊之間的重疊字數 (預設: 20)
   - **實體標準化**: 統一相似實體的名稱 (建議啟用)
   - **關係推理**: 推斷隱含的關係 (建議啟用)

3. **開始處理**
   - 點擊「開始生成知識圖譜」按鈕
   - 查看實時處理進度
   - 處理時間取決於文本長度和 LLM 速度

4. **查看結果**
   - 處理完成後會顯示統計信息
   - 點擊「查看知識圖譜」在新窗口打開視覺化
   - 點擊「下載 HTML」保存結果

## 🎯 適用場景

### 📚 小說分析
上傳小說文本，生成人物關係圖譜：
- 人物之間的關係
- 事件和地點的連接
- 情節線索的可視化

### 📄 學術文獻
分析研究論文或學術材料：
- 概念之間的關係
- 理論框架的結構
- 研究方法的關聯

### 📰 新聞報導
從新聞文章中提取知識：
- 事件時間線
- 人物和組織關係
- 主題關聯

## ⚙️ API 端點

### POST /api/upload
上傳文件並創建處理任務

**參數:**
- `file`: 文件 (multipart/form-data)
- `chunk_size`: 文本塊大小 (可選)
- `overlap`: 重疊字數 (可選)
- `enable_standardization`: 啟用標準化 (可選)
- `enable_inference`: 啟用推理 (可選)

**返回:**
```json
{
  "task_id": "uuid",
  "filename": "example.txt",
  "status": "queued"
}
```

### WebSocket /ws/{task_id}
連接 WebSocket 接收實時進度更新

**消息格式:**
```json
{
  "phase": "extraction",
  "message": "Extracting knowledge...",
  "progress": 50.0,
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/task/{task_id}
獲取任務狀態

### GET /api/view/{task_id}
在瀏覽器中查看生成的知識圖譜

### GET /api/download/{task_id}
下載生成的 HTML 文件

## 🔧 技術架構

### 後端
- **FastAPI** - 現代 Python Web 框架
- **WebSocket** - 實時雙向通信
- **AsyncIO** - 異步處理

### 前端
- **原生 JavaScript** - 無需額外框架
- **WebSocket API** - 實時進度更新
- **響應式設計** - 適配各種螢幕尺寸

### 處理流程
1. 文件上傳到臨時目錄
2. 創建處理任務並返回 task_id
3. 客戶端通過 WebSocket 連接
4. 後端異步處理文本並發送進度更新
5. 生成 HTML 視覺化和 JSON 數據
6. 返回完成狀態和統計信息

## 📝 注意事項

1. **文件大小限制**
   - 建議單個文件不超過 10MB
   - 大文件會增加處理時間

2. **LLM 配置**
   - 確保 LLM 服務正在運行
   - 檢查 API endpoint 是否正確
   - 確認有足夠的 API 配額

3. **處理時間**
   - 取決於文本長度和 chunk 數量
   - 啟用推理會增加處理時間
   - 本地 LLM 通常比雲端 API 慢

4. **瀏覽器兼容性**
   - 建議使用現代瀏覽器 (Chrome, Firefox, Edge, Safari)
   - 需要支持 WebSocket

## 🐛 故障排除

### 無法連接到服務器
- 檢查服務器是否正在運行
- 確認端口 8000 沒有被占用
- 檢查防火牆設置

### 上傳失敗
- 檢查文件格式 (只支持 .txt 和 .md)
- 確認文件編碼為 UTF-8
- 檢查文件大小

### 處理失敗
- 檢查 `config.toml` 配置
- 確認 LLM 服務正常運行
- 查看控制台錯誤日誌

### WebSocket 連接失敗
- 確認瀏覽器支持 WebSocket
- 檢查代理或防火牆設置
- 嘗試刷新頁面重新連接

## 🎨 自定義

### 修改界面樣式
編輯 `web/templates/index.html` 中的 CSS 樣式

### 調整處理參數
修改 `config.toml` 中的默認值

### 添加新功能
擴展 `src/knowledge_graph/api/app.py` 添加新的 API 端點

## 📚 更多資訊

- [主要 README](README.md) - 項目總體介紹
- [配置說明](config.toml) - 詳細配置選項
- [API 文檔](http://localhost:8000/docs) - FastAPI 自動生成的 API 文檔 (啟動服務器後訪問)

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

與主項目相同
