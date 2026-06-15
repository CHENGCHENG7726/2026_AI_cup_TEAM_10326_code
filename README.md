# 2026_AI_cup_TEAM_10326_code

程式碼提供一套基於 PyTorch 實作的深度學習預測基準程式碼，模型採用具備因果遮罩（Causal Mask）的 Transformer 架構，針對球拍運動的拉鋸戰（Rally）數據進行多任務學習（Multi-Task Learning），能同時預測下一拍動作（Action）、下一拍落點（Point）以及最終發球方是否得分（Rally Winner）。

---

## 一、 安裝與配置環境

本程式推薦於 **Google Colaboratory (Colab)** 環境中執行，並建議掛載 **NVIDIA T4 GPU** 以加速訓練。

### 1. 系統需求
* 作業系統：Linux (Colab 預設) 或 Windows 10/11
* Python 版本：3.8 或以上
* 運算資源：強烈建議使用支援 CUDA 之 GPU (至少 8GB VRAM)

### 2. 套件依賴 (Dependencies)
請確保您的環境中已安裝以下 Python 套件。您可以使用 `pip` 進行安裝：

```bash
pip install torch numpy pandas scikit-learn


主要套件版本參考：

torch >= 1.12.0

pandas >= 1.3.0

numpy >= 1.21.0

scikit-learn >= 1.0.0
```

##  二、 檔案目錄與資料準備
請將提供的三個 CSV 檔案與主程式碼放置於同一目錄下：

```
Plaintext
├── main.py                     # 主訓練與推論程式碼
├── train.csv                   # 訓練集資料 (主辦方提供)
├── test.csv                    # 測試集資料 (主辦方提供)
├── sample_submission.csv       # 繳交格式範例檔 (主辦方提供)
└── README.md                   # 本說明檔
```

## 三、 重要模塊說明 (I/O架構)
為了方便第三方除錯與後續二次開發，以下列出系統核心模塊的輸入（Input）與輸出（Output）定義：

1. 特徵工程模塊 add_features(df)
-Input: 原始的 Pandas DataFrame（包含基礎的 sex, score, strikeNumber 等特徵）。

-Output: 擴增後的 DataFrame。新增特徵包含：分數差 (scoreDiff_cat)、關鍵分狀態 (is_critical_point)、發球/拉鋸階段、以及透過 shift 產生的歷史動作轉移特徵 (prev_actionId, action_transition 等)。

2. 資料集模塊 RallyDataset
-Input: 特徵張量 X (形狀為 [N, MAXLEN, num_features])、標籤張量 yA, yP, yR 以及有效序列長度 L。

-Output: 透過 PyTorch DataLoader 產生 Batch，供模型訓練迭代使用。支援序列長度動態對齊與遮罩機制（Padding & Masking）。

3. 核心神經網路 MultiTaskTransformer
-Input: * X: 批次編碼後的特徵張量 (batch_size, seq_len, num_features)。

-lengths: 該批次中每筆資料的實際有效長度 (batch_size,)。

-Output (多任務輸出): * la: 動作預測 logits (batch_size, seq_len, n_act)。

-lp_main: 主落點預測 logits (batch_size, seq_len, n_pt)。

-lp_side: 落點左右輔助預測 logits (batch_size, seq_len, 4)。

-lp_depth: 落點前後輔助預測 logits (batch_size, seq_len, 4)。

-lr: 拉鋸戰勝負預測 logits (batch_size,)（透過 Attention Pooling 結合全局特徵輸出）。

4. 損失函數模塊 FocalLoss
-Input: 模型的預測 logits 以及真實標籤 targets。

-Output: 純量 Loss 值。本模塊專門處理 pointId 等極端類別不平衡問題，並透過傳入的 weight 參數進行動態類別權重調整。

## 四、 重新訓練

1. 執行一鍵訓練與推論
打開終端機（或 Colab 的儲存格），在檔案所在目錄下執行以下指令：

```
Bash
python main.py --epochs 50 --batch 64 --emb 24 --hidden 128 --layers 1 --lr 0.0002
```

2. 參數說明 (Arguments)
使用者可透過命令列引數調整模型超參數，方便進行除錯與消融實驗：

```
--train: 訓練集檔名 (預設 train.csv)

--test: 測試集檔名 (預設 test.csv)

--sample: 提交範本檔名 (預設 sample_submission.csv)

--out: 最終輸出的預測檔名 (預設 submission_lstm_baseline.csv)

--epochs: 訓練迭代次數 (預設 50)

--batch: 批次大小 (預設 64)

--emb: Embedding 維度 (預設 24)

--hidden: Transformer 隱藏層大小 (預設 128)

--layers: Transformer 編碼器層數 (預設 1)

--drop: Dropout 機率 (預設 0.15)

--lr: 初始學習率 (預設 2e-4)
```
