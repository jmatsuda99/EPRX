# 需給調整市場：最小IRR/NPV計算（v10：落札率モデル修正）

## 変更点（v10）
落札率モデルを以下に修正：
- 最初のX年は落札率 α%（任意の初期値）
- その後、前年比でY%ずつ落札率が低下

## 落札率
年 t=1..N に対して
- t <= X: alpha_t = alpha0
- t > X:  alpha_t = alpha0 * (1 - Y)^(t - X)

（alpha0=α/100、Y=Y%/100）

## 既存要素
- β：出力(kW)に乗算（約定量ディレーティング）
- γ：参加日数に乗算（稼働率/運用制約）
- O&M：名目出力(kW)ベースで年次コスト
- 廃止措置：最終年にCAPEX比で一括計上
- RA+AC手数料：その年の総収入に対する割合

## 実行方法
```bash
pip install -r requirements.txt
streamlit run app.py
```
