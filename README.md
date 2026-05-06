# Multivariate Glucodensity Calculator 2nd

**神戸大学臨床糖尿病グループ**

> 利用は自己責任でお願いします。

CGM時系列データから、以下をオンラインで計算する Streamlit アプリです。

- 血糖分布 G(t) の glucodensity 指標
- 速度分布 dG/dt の指標
- 加速度分布 d²G/dt² の指標
- 100点分位関数
- 各患者ごとの 2変量周辺密度関数のカラー等高線図（学会向け配色）
  - 血糖値 vs 速度
  - 血糖値 vs 加速度
  - 速度 vs 加速度
- 各患者図の PNG 保存
- 各患者図の PDF 保存
- **解析対象者全員分の図をPDFで一括ダウンロード**
- **全患者の全図を一括 ZIP 保存**
- 入力テンプレートExcelのダウンロード
- 結果Excel

## 入力形式

Excel または CSV。

- 1行 = 1被検者
- 1列 = 患者ID
- 2列目以降 = 5分ごとのCGM血糖値 mg/dL

例：

| Patient_ID | t0001 | t0002 | t0003 | ... |
|---|---:|---:|---:|---:|
| CGM_01 | 102 | 105 | 108 | ... |
| CGM_02 | 98 | 97 | 101 | ... |

## ローカル実行

```bash
cd glucodensity_streamlit_app
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Streamlit Community Cloudへの公開

1. GitHubに新規リポジトリを作成
2. `app.py` と `requirements.txt` をアップロード
3. Streamlit Community Cloudでそのリポジトリを指定
4. Main file path を `app.py` にする

## 注意

このアプリは Matabuena らの multivariate glucodensity の概念に基づく特徴量抽出ツールです。論文中の Model 1〜6、long-term HbA1c/FPG 予測、mgcv::gam による scalar-on-distribution regression そのものは含みません。
