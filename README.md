# 需給調整市場：最小IRR/NPV計算（v11：アクセスカウンター追加）

## 追加点（v11）
- アクセスカウンター（Total）を追加
  - Streamlitセッションごとに1回だけカウント
  - `access_counter.json` に合計値を保存（同一ディレクトリ）
  - 共有環境での同時アクセスに備え、ベストエフォートでファイルロックを利用

## 注意
- Streamlit Community Cloud 等、実行環境のファイルシステムが「エフェメラル（再起動で消える）」場合、
  カウンターはリセットされる可能性があります。
- 永続化したい場合は、SQLite / Redis / Firestore 等の外部ストレージに置き換えてください。

## 実行方法
```bash
pip install -r requirements.txt
streamlit run app.py
```
