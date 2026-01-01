# Bitcoin Prediction System 

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Data Collection Flow](#data-collection-flow)
3. [Feature Engineering Pipeline](#feature-engineering-pipeline)
4. [Model Training Flow](#model-training-flow)
5. [Prediction Pipeline](#prediction-pipeline)
6. [Service Lifecycle](#service-lifecycle)
7. [Dashboard Architecture](#dashboard-architecture)
8. [Continuous Learning Flow](#continuous-learning-flow)
9. [Error Handling & Recovery](#error-handling--recovery)
10. [File Dependencies](#file-dependencies)

---

## System Architecture Overview

### High-Level System Architecture

```mermaid
graph TB
    subgraph "External Data Sources"
        EXT1[Binance WebSocket<br/>Trade Stream]
        EXT2[Binance WebSocket<br/>Order Book Depth]
        EXT3[RSS News Feeds<br/>6 Sources]
        EXT4[Binance REST API<br/>Historical Data]
    end
    
    subgraph "Data Collection Layer"
        DC1[live_stream.py<br/>Trade Collector]
        DC2[aggregate_live_to_candles.py<br/>Candle Aggregator]
        DC3[sentiment_ingest.py<br/>News Sentiment]
        DC4[orderbook_depth.py<br/>Market Depth]
        DC5[historical_data.py<br/>Historical Fetcher]
    end
    
    subgraph "Data Storage Layer"
        DS1[(btc_trades_live.csv<br/>Raw Trades)]
        DS2[(btc_live_candles.csv<br/>1-min Candles)]
        DS3[(sentiment_events.csv<br/>sentiment_minute.csv)]
        DS4[(orderbook_depth.csv<br/>Market Microstructure)]
        DS5[(btc_historical_clean.csv<br/>60 Days History)]
    end
    
    subgraph "Processing Layer"
        PR1[data_cleaner.py<br/>Outlier Removal]
        PR2[build_dataset.py<br/>Feature Engineering]
    end
    
    subgraph "Feature Storage"
        FS1[(btc_features_normalized.csv<br/>25 Features)]
        FS2[(scalers.pkl<br/>Normalization)]
    end
    
    subgraph "ML Layer"
        ML1[train_models.py<br/>Initial Training]
        ML2[continuous_learning.py<br/>Hourly Retraining]
    end
    
    subgraph "Model Storage"
        MS1[(LSTM Model<br/>.pth)]
        MS2[(Transformer Model<br/>.pth)]
        MS3[(Random Forest<br/>.pkl)]
    end
    
    subgraph "Presentation Layer"
        UI1[binance_dashboard.py<br/>Streamlit Dashboard]
        UI2[orchestrate_pipeline.py<br/>Service Manager]
    end
    
    EXT1 --> DC1
    EXT2 --> DC4
    EXT3 --> DC3
    EXT4 --> DC5
    
    DC1 --> DS1
    DC2 --> DS2
    DC3 --> DS3
    DC4 --> DS4
    DC5 --> DS5
    
    DS1 --> DC2
    DS5 --> PR1
    PR1 --> PR2
    DS2 --> PR2
    DS3 --> PR2
    
    PR2 --> FS1
    PR2 --> FS2
    
    FS1 --> ML1
    FS2 --> ML1
    FS1 --> ML2
    FS2 --> ML2
    
    ML1 --> MS1
    ML1 --> MS2
    ML1 --> MS3
    ML2 --> MS1
    ML2 --> MS2
    ML2 --> MS3
    
    MS1 --> UI1
    MS2 --> UI1
    MS3 --> UI1
    DS2 --> UI1
    DS3 --> UI1
    DS4 --> UI1
    FS2 --> UI1
    
    UI2 -.manages.-> DC1
    UI2 -.manages.-> DC2
    UI2 -.manages.-> DC3
    UI2 -.manages.-> DC4
    UI2 -.manages.-> ML2
    UI2 -.manages.-> UI1
    
    style EXT1 fill:#e1f5ff
    style EXT2 fill:#e1f5ff
    style EXT3 fill:#e1f5ff
    style EXT4 fill:#e1f5ff
    style DS1 fill:#fff3e0
    style DS2 fill:#fff3e0
    style DS3 fill:#fff3e0
    style DS4 fill:#fff3e0
    style DS5 fill:#fff3e0
    style FS1 fill:#f3e5f5
    style FS2 fill:#f3e5f5
    style MS1 fill:#e8f5e9
    style MS2 fill:#e8f5e9
    style MS3 fill:#e8f5e9
    style UI1 fill:#fce4ec
```

---

## Data Collection Flow

### Live Trade Streaming (live_stream.py)

```mermaid
flowchart TD
    Start([Start Service]) --> Init[Initialize WebSocket<br/>Connection]
    Init --> Connect{Connect to<br/>Binance?}
    
    Connect -->|Success| Listen[Listen for<br/>Trade Events]
    Connect -->|Fail| Backoff[Exponential<br/>Backoff]
    Backoff --> Wait[Wait 1s → 60s]
    Wait --> Connect
    
    Listen --> Receive[Receive Trade<br/>Message]
    Receive --> Parse[Parse JSON<br/>Extract Fields]
    Parse --> Buffer[Add to<br/>Memory Buffer]
    
    Buffer --> CheckBatch{Batch Timer<br/>2 seconds?}
    CheckBatch -->|No| Listen
    CheckBatch -->|Yes| Flush[Flush Buffer<br/>to CSV]
    
    Flush --> Append[Append to<br/>btc_trades_live.csv]
    Append --> Clear[Clear Buffer]
    Clear --> Listen
    
    Listen --> Error{Connection<br/>Lost?}
    Error -->|Yes| Reconnect[Auto-Reconnect]
    Reconnect --> Backoff
    Error -->|No| Listen
    
    style Start fill:#4caf50
    style Append fill:#ff9800
    style Error fill:#f44336
```

### Candle Aggregation (aggregate_live_to_candles.py)

```mermaid
flowchart TD
    Start([Start Aggregator]) --> Loop[Main Loop<br/>Every 1 Second]
    
    Loop --> CheckRotate{2 Hours<br/>Passed?}
    CheckRotate -->|Yes| Rotate[Rotate CSV<br/>Keep Last 200k]
    CheckRotate -->|No| ReadTail
    Rotate --> ReadTail[Read Last 100KB<br/>of Trades CSV]
    
    ReadTail --> Parse[Parse Trades<br/>to DataFrame]
    Parse --> Floor[Floor Time<br/>to Minutes]
    Floor --> Filter[Filter Out<br/>Current Minute]
    
    Filter --> CheckComplete{Complete<br/>Minutes?}
    CheckComplete -->|No| Wait[Wait 1s]
    CheckComplete -->|Yes| Aggregate[Group by Minute<br/>Calculate OHLCV]
    
    Aggregate --> Merge[Merge with<br/>Existing Candles]
    Merge --> Dedupe[Remove<br/>Duplicates]
    Dedupe --> Keep[Keep Last<br/>500 Candles]
    Keep --> Save[Save to<br/>btc_live_candles.csv]
    
    Save --> Wait
    Wait --> Loop
    
    style Start fill:#4caf50
    style Rotate fill:#2196f3
    style Save fill:#ff9800
```

### Sentiment Collection (sentiment_ingest.py)

```mermaid
flowchart TD
    Start([Start Service]) --> Init[Initialize<br/>VADER Analyzer]
    Init --> Loop[Main Loop<br/>Every 60 Seconds]
    
    Loop --> Fetch[Fetch from<br/>6 RSS Feeds]
    Fetch --> ParseRSS[Parse RSS<br/>XML Entries]
    
    ParseRSS --> CheckLang{English<br/>Language?}
    CheckLang -->|No| Skip1[Skip Entry]
    CheckLang -->|Yes| CheckDupe{Duplicate<br/>Hash?}
    
    CheckDupe -->|Yes| Skip2[Skip Entry]
    CheckDupe -->|No| CalcRel[Calculate<br/>Relevance Score]
    
    CalcRel --> CheckRel{Relevance<br/>> 0.1?}
    CheckRel -->|No| Skip3[Skip Entry]
    CheckRel -->|Yes| Sentiment[VADER Sentiment<br/>Analysis]
    
    Sentiment --> SaveEvent[Save to<br/>sentiment_events.csv]
    SaveEvent --> Buffer[Add to<br/>Minute Buffer]
    
    Buffer --> CheckNext{More<br/>Entries?}
    CheckNext -->|Yes| ParseRSS
    CheckNext -->|No| Aggregate[Aggregate<br/>Minute Data]
    
    Aggregate --> SaveMinute[Save to<br/>sentiment_minute.csv]
    SaveMinute --> Wait[Wait 60s]
    Wait --> Loop
    
    Skip1 --> CheckNext
    Skip2 --> CheckNext
    Skip3 --> CheckNext
    
    style Start fill:#4caf50
    style Sentiment fill:#9c27b0
    style SaveEvent fill:#ff9800
    style SaveMinute fill:#ff9800
```

---

## Feature Engineering Pipeline

### Complete Feature Engineering Flow

```mermaid
flowchart TD
    Start([build_dataset.py<br/>Triggered]) --> LoadHist[Load Historical<br/>btc_historical_clean.csv]
    LoadHist --> LoadLive[Load Live Candles<br/>btc_live_candles.csv]
    
    LoadLive --> Merge1[Merge Historical<br/>+ Live]
    Merge1 --> Dedupe[Remove Duplicates<br/>by timeOpen]
    Dedupe --> Sort[Sort by<br/>timeOpen]
    
    Sort --> TechFeatures[Create Technical<br/>Features]
    
    TechFeatures --> Lag[Lag Features<br/>lag_1, lag_5, lag_15]
    Lag --> Returns[Log Returns<br/>log_return]
    Returns --> MA[Moving Averages<br/>ma_5, ma_15, ma_60]
    MA --> Vol[Volatility<br/>vol_5, vol_15, vol_60]
    Vol --> RSI[RSI Indicator<br/>rsi_14]
    RSI --> MACD[MACD<br/>line, signal, histogram]
    MACD --> BB[Bollinger Bands<br/>upper, middle, lower]
    BB --> EMA[EMA<br/>ema_12, ema_26]
    
    EMA --> LoadSent[Load Sentiment<br/>sentiment_minute.csv]
    LoadSent --> MergeSent[Merge on<br/>timeOpen]
    
    MergeSent --> ForwardFill[Forward Fill<br/>Missing Sentiment]
    ForwardFill --> Features25[25 Features<br/>Complete]
    
    Features25 --> Split[Split Train/Val<br/>80% / 20%]
    Split --> FitScalers[Fit MinMaxScaler<br/>on Train Data]
    FitScalers --> Transform[Transform<br/>All Data]
    
    Transform --> SaveFeatures[Save<br/>btc_features_normalized.csv]
    SaveFeatures --> SaveScalers[Save<br/>scalers.pkl]
    SaveScalers --> End([Complete])
    
    style Start fill:#4caf50
    style TechFeatures fill:#2196f3
    style LoadSent fill:#9c27b0
    style SaveFeatures fill:#ff9800
    style SaveScalers fill:#ff9800
    style End fill:#4caf50
```

### Feature Calculation Details

```mermaid
graph LR
    subgraph "Technical Indicators (20)"
        T1[Price Lags<br/>3 features]
        T2[Returns<br/>1 feature]
        T3[Moving Averages<br/>3 features]
        T4[Volatility<br/>3 features]
        T5[RSI<br/>1 feature]
        T6[MACD<br/>3 features]
        T7[Bollinger Bands<br/>3 features]
        T8[EMA<br/>2 features]
    end
    
    subgraph "Sentiment Features (5)"
        S1[sentiment_mean<br/>Average sentiment]
        S2[sentiment_neg_mean<br/>Negative score]
        S3[relevance_score<br/>News relevance]
        S4[events_count<br/>Event volume]
        S5[negative_spike_flag<br/>Fear indicator]
    end
    
    subgraph "Final Dataset"
        F[25 Features<br/>Normalized 0-1]
    end
    
    T1 --> F
    T2 --> F
    T3 --> F
    T4 --> F
    T5 --> F
    T6 --> F
    T7 --> F
    T8 --> F
    S1 --> F
    S2 --> F
    S3 --> F
    S4 --> F
    S5 --> F
    
    style T1 fill:#bbdefb
    style T2 fill:#bbdefb
    style T3 fill:#bbdefb
    style T4 fill:#bbdefb
    style T5 fill:#bbdefb
    style T6 fill:#bbdefb
    style T7 fill:#bbdefb
    style T8 fill:#bbdefb
    style S1 fill:#e1bee7
    style S2 fill:#e1bee7
    style S3 fill:#e1bee7
    style S4 fill:#e1bee7
    style S5 fill:#e1bee7
    style F fill:#c8e6c9
```

---

## Model Training Flow

### Initial Training Process (train_models.py)

```mermaid
flowchart TD
    Start([train_models.py<br/>Started]) --> Load[Load Normalized<br/>Features CSV]
    Load --> CreateSeq[Create Sequences<br/>60 timesteps]
    
    CreateSeq --> Split[Split Data<br/>80% Train / 20% Val]
    Split --> Baseline{Train<br/>Baseline?}
    
    Baseline -->|Yes| RF[Random Forest<br/>Regression]
    RF --> SaveRF[Save<br/>btc_model_reg.pkl]
    
    Baseline -->|No| Deep
    SaveRF --> Deep{Train Deep<br/>Learning?}
    
    Deep -->|Yes| CheckGPU{GPU<br/>Available?}
    CheckGPU -->|Yes| UseGPU[Use CUDA]
    CheckGPU -->|No| UseCPU[Use CPU]
    
    UseGPU --> InitLSTM[Initialize LSTM<br/>2 layers, 64 units]
    UseCPU --> InitLSTM
    
    InitLSTM --> TrainLSTM[Train LSTM<br/>10 epochs, early stop]
    TrainLSTM --> SaveLSTM[Save<br/>btc_model_reg.pth]
    
    SaveLSTM --> InitTrans[Initialize Transformer<br/>2 layers, 4 heads]
    InitTrans --> TrainTrans[Train Transformer<br/>10 epochs, early stop]
    TrainTrans --> SaveTrans[Save Alternative<br/>Model]
    
    SaveTrans --> Evaluate[Evaluate All<br/>Models]
    Evaluate --> Compare[Compare<br/>Performance]
    Compare --> SelectBest[Select Best<br/>Model]
    SelectBest --> Report[Generate<br/>Metrics Report]
    
    Deep -->|No| Report
    Report --> End([Training<br/>Complete])
    
    style Start fill:#4caf50
    style RF fill:#ff9800
    style InitLSTM fill:#2196f3
    style InitTrans fill:#9c27b0
    style End fill:#4caf50
```

### Model Architecture Details

```mermaid
graph TB
    subgraph "LSTM Architecture"
        L1[Input Layer<br/>60 timesteps × 25 features]
        L2[LSTM Layer 1<br/>64 hidden units<br/>20% dropout]
        L3[LSTM Layer 2<br/>64 hidden units<br/>20% dropout]
        L4[Take Last Timestep<br/>64 features]
        L5[Fully Connected<br/>64 → 1]
        L6[Output<br/>Predicted Return]
        
        L1 --> L2
        L2 --> L3
        L3 --> L4
        L4 --> L5
        L5 --> L6
    end
    
    subgraph "Transformer Architecture"
        T1[Input Layer<br/>60 timesteps × 25 features]
        T2[Embedding<br/>25 → 64 dimensions]
        T3[Transformer Encoder<br/>2 layers, 4 heads]
        T4[Take Last Timestep<br/>64 features]
        T5[Fully Connected<br/>64 → 1]
        T6[Output<br/>Predicted Return]
        
        T1 --> T2
        T2 --> T3
        T3 --> T4
        T4 --> T5
        T5 --> T6
    end
    
    subgraph "Random Forest"
        R1[Input<br/>25 features<br/>last row only]
        R2[100 Decision Trees<br/>max_depth=10]
        R3[Ensemble Vote<br/>Average predictions]
        R4[Output<br/>Predicted Return]
        
        R1 --> R2
        R2 --> R3
        R3 --> R4
    end
    
    style L1 fill:#e3f2fd
    style L6 fill:#c8e6c9
    style T1 fill:#f3e5f5
    style T6 fill:#c8e6c9
    style R1 fill:#fff3e0
    style R4 fill:#c8e6c9
```

---

## Prediction Pipeline

### Real-Time Prediction Flow

```mermaid
flowchart TD
Start([User Opens<br/>Dashboard]) --> Orchestrator[Data Orchestrator<br/>Background Thread]
Orchestrator --> FetchPrice[Fetch Current Price<br/>Binance API]
Orchestrator --> LoadCandles[Load Last 200<br/>Candles]
Orchestrator --> LoadModels[Load Trained<br/>Models]
Orchestrator --> LoadScalers[Load<br/>Scalers]
FetchPrice --> Cache
LoadCandles --> CreateTech[Create Technical<br/>Features]
CreateTech --> LoadSent[Load Sentiment<br/>Features]
LoadSent --> MergeFeat[Merge Features<br/>on timeOpen]
MergeFeat --> Normalize[Normalize with<br/>Scalers]
Normalize --> CreateSeq[Create 60-step<br/>Sequence]
LoadModels --> SelectModel{Which<br/>Model?}
SelectModel -->|LSTM| LSTM[LSTM Inference]
SelectModel -->|Transformer| Trans[Transformer Inference]
SelectModel -->|RF| RF[Random Forest Predict]
CreateSeq --> LSTM
CreateSeq --> Trans
CreateSeq --> RF
LSTM --> Return[Predicted<br/>Return]
Trans --> Return
RF --> Return
LoadScalers --> Normalize
Return --> CalcPrice[Calculate Price<br/>current multiplied by return]
CalcPrice --> Direction{Price ><br/>Current?}
Direction -->|Yes| Up[Direction: UP 🟢]
Direction -->|No| Down[Direction: DOWN 🔴]
Up --> Cache[Update<br/>GLOBAL_CACHE]
Down --> Cache
Cache --> UI[UI Fragments<br/>Read Cache]
UI --> Display[Display to<br/>User]
Display --> Wait[Wait 1s]
Wait --> Orchestrator
style Start fill:#4caf50
style LSTM fill:#2196f3
style Trans fill:#9c27b0
style RF fill:#ff9800
style Display fill:#f44336
```

---

## Service Lifecycle

### System Startup Sequence

```mermaid
sequenceDiagram
    participant User
    participant Dashboard as binance_dashboard.py
    participant Setup as Setup Thread
    participant LiveStream as live_stream.py
    participant Aggregator as aggregate_live_to_candles.py
    participant Sentiment as sentiment_ingest.py
    participant OrderBook as orderbook_depth.py
    participant Training as train_models.py
    participant CL as continuous_learning.py
    
    User->>Dashboard: Start Dashboard
    Dashboard->>Setup: Start Background Setup
    
    alt fresh_start = true
        Setup->>Setup: Delete All Data & Models
    end
    
    Setup->>LiveStream: Start WebSocket Stream
    Setup->>Aggregator: Start Candle Aggregator
    
    alt No Dataset Exists
        Setup->>Setup: Download Historical Data (60 days)
        Setup->>Setup: Clean Data
        Setup->>Setup: Build Dataset (25 features)
    end
    
    alt No Models Exist
        Setup->>Training: Train All Models
        Training-->>Setup: Models Saved
    end
    
    Setup->>CL: Start Continuous Learning
    Setup->>Sentiment: Start Sentiment Collection
    Setup->>OrderBook: Start Order Book Collector
    
    Setup-->>Dashboard: Setup Complete
    Dashboard-->>User: Show Dashboard
    
    loop Every Second
        Dashboard->>Dashboard: Update UI from Cache
    end
    
    loop Every 60 Seconds
        Sentiment->>Sentiment: Fetch & Score News
    end
    
    loop Every Hour
        CL->>CL: Check for Retraining
    end
```

### Service Dependencies

```mermaid
graph TD
    subgraph "Critical Path"
        A[live_stream.py<br/>MUST START FIRST]
        B[aggregate_live_to_candles.py<br/>Depends on Trades]
        C[build_dataset.py<br/>Depends on Candles]
        D[train_models.py<br/>Depends on Dataset]
        E[binance_dashboard.py<br/>Depends on Models]
        
        A --> B
        B --> C
        C --> D
        D --> E
    end
    
    subgraph "Independent Services"
        F[sentiment_ingest.py<br/>Independent]
        G[orderbook_depth.py<br/>Independent]
        H[continuous_learning.py<br/>Independent]
    end
    
    F -.optional.-> C
    G -.optional.-> E
    H -.updates.-> D
    
    style A fill:#f44336
    style B fill:#ff9800
    style C fill:#ffc107
    style D fill:#4caf50
    style E fill:#2196f3
    style F fill:#9c27b0
    style G fill:#9c27b0
    style H fill:#9c27b0
```

---

## Dashboard Architecture

### Dashboard Component Structure

```mermaid
graph TB
    subgraph "Main Thread"
        Main[main Function]
        Main --> CSS[Inject Custom CSS]
        Main --> Orchestrator[Start Data Orchestrator]
        Main --> Services[Start Background Services]
    end
    
    subgraph "Background Orchestrator Thread"
        Orch[Orchestrator Loop<br/>Every 1 Second]
        Orch --> API[Fetch Binance API<br/>24h Ticker]
        Orch --> CSV[Load CSV Files<br/>Candles, Sentiment, OB]
        Orch --> Pred[Generate Prediction<br/>LSTM/Transformer/RF]
        
        API --> Cache[GLOBAL_CACHE]
        CSV --> Cache
        Pred --> Cache
    end
    
    subgraph "UI Fragments (Auto-Refresh)"
        F1[Header Ticker<br/>1s refresh]
        F2[TradingView Chart<br/>30s refresh]
        F3[Sentiment Panel<br/>5s refresh]
        F4[Order Book<br/>1s refresh]
        F5[Recent Trades<br/>2s refresh]
        
        Cache --> F1
        Cache --> F2
        Cache --> F3
        Cache --> F4
        Cache --> F5
    end
    
    subgraph "User Interface"
        UI[Streamlit UI<br/>Rendered]
    end
    
    F1 --> UI
    F2 --> UI
    F3 --> UI
    F4 --> UI
    F5 --> UI
    
    style Main fill:#2196f3
    style Orch fill:#4caf50
    style Cache fill:#ff9800
    style UI fill:#f44336
```

### Dashboard Data Flow

```mermaid
flowchart LR
    subgraph "Data Sources"
        S1[(CSV Files)]
        S2[Binance API]
        S3[(JSON Files)]
        S4[(Model Files)]
    end
    
    subgraph "Background Thread"
        BG[Data Orchestrator<br/>Runs Every 1s]
    end
    
    subgraph "Shared Memory"
        CACHE[GLOBAL_CACHE<br/>Dictionary]
    end
    
    subgraph "UI Layer"
        UI1[Header]
        UI2[Chart]
        UI3[Sentiment]
        UI4[Order Book]
        UI5[Trades]
    end
    
    S1 --> BG
    S2 --> BG
    S3 --> BG
    S4 --> BG
    
    BG --> CACHE
    
    CACHE --> UI1
    CACHE --> UI2
    CACHE --> UI3
    CACHE --> UI4
    CACHE --> UI5
    
    style BG fill:#4caf50
    style CACHE fill:#ff9800
```

---

## Continuous Learning Flow

### Continuous Learning Process

```mermaid
flowchart TD
    Start([continuous_learning.py<br/>Started]) --> Init[Initialize<br/>Candle Counter = 0]
    
    Init --> Loop[Main Loop<br/>Every 60 Seconds]
    Loop --> Check[Check for New<br/>Candles]
    
    Check --> Compare{New Candles<br/>Found?}
    Compare -->|No| Wait[Wait 60s]
    Compare -->|Yes| LoadNew[Load New<br/>Candles]
    
    LoadNew --> Append[Append to<br/>Historical CSV]
    Append --> UpdateFeatures[Update Features<br/>Incrementally]
    UpdateFeatures --> UpdateScalers[Update Scalers<br/>Incrementally]
    UpdateScalers --> Increment[Increment<br/>Candle Counter]
    
    Increment --> CheckCount{Counter<br/>>= 60?}
    CheckCount -->|No| Wait
    CheckCount -->|Yes| Backup[Backup Current<br/>Models]
    
    Backup --> Retrain[Trigger Retraining<br/>train_models.py]
    Retrain --> WaitTrain[Wait for<br/>Training Complete]
    WaitTrain --> LoadNew2[Load New<br/>Models]
    LoadNew2 --> Reset[Reset Counter<br/>to 0]
    Reset --> Wait
    
    Wait --> Loop
    
    style Start fill:#4caf50
    style Retrain fill:#f44336
    style LoadNew2 fill:#2196f3
```

### Retraining Timeline

```mermaid
gantt
    title Continuous Learning Timeline
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Data Collection
    Live Trades Streaming     :active, 00:00, 120m
    Candle Aggregation        :active, 00:00, 120m
    
    section Feature Updates
    New Candle 1              :milestone, 00:01, 0m
    New Candle 30             :milestone, 00:30, 0m
    New Candle 60             :milestone, 01:00, 0m
    
    section Retraining
    Trigger Retraining        :crit, 01:00, 10m
    Model Training            :crit, 01:00, 10m
    Load New Models           :crit, 01:10, 1m
    
    section Next Cycle
    Continue Collection       :active, 01:11, 49m
    Next Retrain              :milestone, 02:00, 0m
```

---

## Error Handling & Recovery

### WebSocket Reconnection Strategy

```mermaid
flowchart TD
    Start([WebSocket<br/>Connected]) --> Listen[Listen for<br/>Messages]
    
    Listen --> Error{Connection<br/>Error?}
    Error -->|No| Listen
    Error -->|Yes| Log[Log Error<br/>Message]
    
    Log --> Init[Initialize<br/>Backoff = 1s]
    Init --> Attempt[Reconnection<br/>Attempt]
    
    Attempt --> Try{Connect<br/>Success?}
    Try -->|Yes| Reset[Reset Backoff<br/>to 1s]
    Reset --> Listen
    
    Try -->|No| Wait[Wait<br/>Backoff Duration]
    Wait --> Double[Double Backoff<br/>Max 60s]
    Double --> CheckMax{Backoff<br/>> 60s?}
    
    CheckMax -->|Yes| Cap[Cap at 60s]
    CheckMax -->|No| Attempt
    Cap --> Attempt
    
    style Start fill:#4caf50
    style Error fill:#f44336
    style Try fill:#ff9800
```

### Data Validation Flow

```mermaid
flowchart TD
    Start([Data Received]) --> CheckNull{Contains<br/>NULL?}
    
    CheckNull -->|Yes| HandleNull[Forward Fill<br/>or Drop]
    CheckNull -->|No| CheckOutlier{Outlier<br/>Detected?}
    
    HandleNull --> CheckOutlier
    CheckOutlier -->|Yes| Remove[Remove or<br/>Interpolate]
    CheckOutlier -->|No| CheckDupe{Duplicate<br/>Entry?}
    
    Remove --> CheckDupe
    CheckDupe -->|Yes| Skip[Skip Entry]
    CheckDupe -->|No| CheckRange{Values in<br/>Range?}
    
    Skip --> End
    CheckRange -->|No| Clip[Clip to<br/>Valid Range]
    CheckRange -->|Yes| Valid[Mark as<br/>Valid]
    
    Clip --> Valid
    Valid --> Save[Save to<br/>Storage]
    Save --> End([Complete])
    
    style Start fill:#4caf50
    style CheckNull fill:#ff9800
    style CheckOutlier fill:#ff9800
    style CheckDupe fill:#ff9800
    style CheckRange fill:#ff9800
    style End fill:#4caf50
```

---

## File Dependencies

### File Dependency Graph

```mermaid
graph TD
    subgraph "Raw Data Files"
        F1[btc_trades_live.csv]
        F2[sentiment_events.csv]
        F3[btc_historical.csv]
    end
    
    subgraph "Processed Data Files"
        F4[btc_live_candles.csv]
        F5[sentiment_minute.csv]
        F6[btc_historical_clean.csv]
        F7[orderbook_depth.csv]
    end
    
    subgraph "Feature Files"
        F8[btc_dataset.csv]
        F9[btc_features.csv]
        F10[btc_features_normalized.csv]
        F11[scalers.pkl]
    end
    
    subgraph "Model Files"
        F12[btc_model_reg.pkl]
        F13[btc_model_reg.pth]
        F14[btc_model_cls.pkl]
    end
    
    subgraph "Configuration"
        F15[config.yaml]
    end
    
    subgraph "Runtime Files"
        F16[latest_orderbook.json]
        F17[logs/*.log]
    end
    
    F1 --> F4
    F2 --> F5
    F3 --> F6
    
    F4 --> F8
    F5 --> F8
    F6 --> F8
    
    F8 --> F9
    F9 --> F10
    F10 --> F11
    
    F10 --> F12
    F10 --> F13
    F10 --> F14
    
    F15 -.configures.-> F1
    F15 -.configures.-> F2
    F15 -.configures.-> F3
    
    style F1 fill:#ffcdd2
    style F2 fill:#ffcdd2
    style F3 fill:#ffcdd2
    style F4 fill:#fff9c4
    style F5 fill:#fff9c4
    style F6 fill:#fff9c4
    style F7 fill:#fff9c4
    style F10 fill:#c8e6c9
    style F11 fill:#c8e6c9
    style F12 fill:#bbdefb
    style F13 fill:#bbdefb
    style F14 fill:#bbdefb
```

### File Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: Service Starts
    Created --> Writing: Data Incoming
    Writing --> Rotating: Size Threshold
    Rotating --> Writing: Trimmed
    Writing --> Reading: Dashboard Access
    Reading --> Writing: Continue Updates
    Writing --> Archived: Cleanup
    Archived --> [*]
    
    note right of Rotating
        btc_trades_live.csv
        Rotates every 2 hours
    end note
    
    note right of Reading
        Dashboard reads
        without blocking writes
    end note
```

---

## Summary

This comprehensive diagram collection covers:

✅ **System Architecture**: Complete overview of all components
✅ **Data Flows**: How data moves through the system
✅ **Process Flows**: Detailed step-by-step operations
✅ **Service Interactions**: How services communicate
✅ **Error Handling**: Recovery mechanisms
✅ **File Dependencies**: Data lineage and relationships

```

```


