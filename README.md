# LLMCafe

VRChatワールド用の接客NPCのスクリプト。NPCは客が入力欄に打った文章を読み、注文の品を席まで運んで、音声で返事をする。

推論に外部のサーバーは使わない。同じインスタンスにいる参加者のうち、入力欄の「ワーカーになる」を押した1人のGPUで動く。以下、この1人をワーカーと呼ぶ。推論には[ShaderGPT](https://gitlab.com/lox9973/ShaderGPT)を使っている。ShaderGPTは言語モデルをピクセルシェーダーで動かすライブラリで、モデルの重みはテクスチャとして持つ。

このリポジトリは、公開しているカフェのワールドからNPCまわりのスクリプトを抜き出したもの。モデルの重み、ShaderGPT本体、アバター、内装は含まない。入れればそのまま動くサンプルではなく、自分のワールドに組み込むときに読んで流用するためのコードになっている。

## 接客の流れ

1. 判断モデルが客の文章を用件ごとに分ける。注文、会計、雑談など
2. 注文なら、会話モデルが品物と届け先を読み取る
3. 判断モデルがNPCの次の行動を選び、NPCが品物を運ぶ
4. 会話モデルが返事を書き、声のモデルとコーデックが音声にする

届けた品物は客ごとの伝票に記録される。会計を頼むと合計を答え、払わずに店を出ようとした客には声をかける。

## 必要なもの

- Unity 2022.3とVRChat SDK（Worlds）、UdonSharp。SDK 3.10.5で動かしている
- ShaderGPT。commit `99bf46ed`に合わせてある
- モデルの変換に使うPython。パッケージはShaderGPTの`Python~/requirements.txt`に加えて、コーデックの変換に`miocodec`が要る
- 下のモデル4つ

| モデル | 用途 | 以下での呼び名 |
|---|---|---|
| LFM2.5-1.2B-JP-202606 Q4_0 | 注文の読み取りと返事 | 会話モデル |
| decider-0.8b Q4_0 | 用件の分類と、NPCの次の行動の選択。文章は書かず、選択肢から1つを選ぶ | 判断モデル |
| MioTTS-0.4B Q4_0 | セリフを音声トークンにする | 声のモデル |
| MioCodec-25Hz-44.1kHz-v2 | 音声トークンを波形に戻す | コーデック |

配布元とリビジョンは`ATTRIBUTION.txt`にある。

## 組み込み

モデルの変換、Bake、シーンへの配置の順に進める。公開しているワールドはこの手順で作ったが、このリポジトリのファイルだけで最初から通す確認はしていない。足りない入力は各手順に書いた。

### 1. スクリプトを置く

`Runtime/CafeNPC`と`Runtime/WorldLLM`をUnityプロジェクトの`Assets`の下に置く。場所は自由。`Editor/`の2ファイルは`Editor`という名前のフォルダに入れる。

`Runtime/ShaderGPT`の4ファイルは、ShaderGPTのフォルダへ同じ相対パスでコピーする。`Udon/GPTGenerator.cs`、`Udon/GPTTokenizer.cs`、`Shader/Reduce.shader`の3つは上書きになり、`Script/Models/MioCodecDecoder.cs`は新しく加わる。

### 2. モデルをテクスチャに変換する

変換に使うスクリプトは`Tools/`にある。変換したフォルダはUnityプロジェクトの下の場所に置く。会話モデルと声のモデルは、置いたフォルダを右クリックして`ShaderGPT/ImportModel`で取り込む。

| モデル | 置くフォルダ |
|---|---|
| 会話モデル | 自由 |
| 判断モデル | `Assets/CafeNPC/Models/Decider08B-Q4` |
| 声のモデル | `Assets/CafeNPC/Models/MioTTS-0.4B-Q4_0` |
| コーデック | `Assets/CafeNPC/Models/MioCodecDecoder-F1up3` |

下の3つのフォルダ名は、`Editor/CafeDeciderBaker.cs`の`WorldFolder`と、`Editor/CafeVoiceInstaller.cs`の`LanguageModel`、`Decoder`に書いてある。場所を変えるならこの定数を直す。

`Tools/`のスクリプトはShaderGPTの`Python~`フォルダを読み込む。場所は環境変数`SHADERGPT_PYTHON`で渡す。

#### 会話モデルと声のモデル

どちらも同じ2段階で変換する。まず語彙を絞る。

```
python prune_text_vocabulary.py <モデルのフォルダ> <元のgguf> <出力先> world
```

`<モデルのフォルダ>`には、そのモデルの`tokenizer.json`、`tokenizer_config.json`、`config.json`を置いておく。出力先には、語彙を絞ったggufとトークナイザができる。これをShaderGPTの`convert_hf_model.py`でテクスチャにする。

```
python convert_hf_model.py <出力先> <テクスチャの出力先> --gguf <絞ったgguf>
```

語彙を絞るのはワールドの容量のため。VRChatにアップロードできたのは約1.2GBまでで、モデルをそのまま入れると収まらなかった。`world`を付けると、公開しているワールドと同じ一覧`Tools/text-vocabulary.json`がそのまま使われる。残るテキストのトークンは17,013個。`world`を付けないと、残すトークンをプロンプトとC#のソースから選び直す。選ぶ元にしたファイルの一部はこのリポジトリにないので、結果は公開しているワールドと一致しない。

会話モデルは配布されているQ4_0のggufでも、トークン埋め込みだけQ6_Kになっている。公開しているワールドでは、語彙を絞る前にllama.cppの`llama-quantize`でここもQ4_0にした。`--token-embedding-type q4_0`を付ける。声のモデルの埋め込みはQ6_Kのまま使う。Q4_0にすると声が崩れた。

#### 判断モデル

```
python decider/export_decider_q4.py Q4_0 <モデルのフォルダ> words
```

入力と出力の場所は決まっていて、変えられない。

- `<モデルのフォルダ>`には、Hugging Faceから取った判断モデル一式と`download-manifest.json`を置く。`download-manifest.json`は`{"repository": "Mapika/decider-0.8b", "revision": "<リビジョン>"}`の形で自分で書く
- ggufは`Tools/decider-export/decider-0.8b-Q4_0.gguf`に置く
- テクスチャは`Tools/decider-export/decider-0.8b-q4_0-textures-vocab-words`にできる。これを上の表のフォルダへ移す
- 同じフォルダに`Tools/decider/decider-label-table.json`も置く。選択肢の記号とトークンの対応表で、ないとBakeが止まる

`words`を付けると語彙も絞る。このとき`decider/prune_vocabulary.py`が呼ばれ、判断モデルと会話モデルのトークナイザを読む。場所は次のどれかで渡す。

- 環境変数`DECIDER_TOKENIZER`と`CONVERSATION_TOKENIZER`に`tokenizer.json`のパスを入れる
- `Tools/decider`に`tokenizer.json`と`conversation-tokenizer.json`の名前で置く
- `Tools/decider`か`Tools`に`paths.local.json`を置いて、そこに書く

```json
{"DECIDER_TOKENIZER": "D:/models/decider-0.8b/tokenizer.json"}
```

#### コーデック

```
python voice/bake_miocodec.py <作業フォルダ> <話者の.pt> <出力先>
python voice/export_miocodec_textures.py <出力先> <テクスチャの出力先>
```

1つ目はデコーダの重みを話者1人に固定して作り直し、2つ目がそれをテクスチャにする。話者の埋め込みは`voice/F1_up3.json`に入っている。スクリプトが読むのは`.pt`なので、先に変換しておく。

```python
import json, torch
torch.save(torch.tensor(json.load(open('F1_up3.json'))['embedding']), 'F1_up3.pt')
```

`<作業フォルダ>`には`download/miocodec-v2/`にMioCodecの`config.yaml`と`model.safetensors`を置く。もう1つ、照合用の音声トークン列`out/unity-sample-1step.json`も読むが、これを作るスクリプトはこのリポジトリに入れていない。このままでは最後まで動かないので、照合の部分を外すか、自分でトークン列を用意する必要がある。

`voice/voices_miotts.py`は話者の埋め込みを作ったときのスクリプトで、参考として入れてある。入力がないので単体では動かない。

### 3. Bakeする

ShaderGPTでは、推論を1回走らせて計算の手順をprefabに書き出しておく。これがBakeで、ワールドで動くのはこのprefabになる。どれもPlay Modeで実行する。

| モデル | 方法 |
|---|---|
| 会話モデル | ShaderGPTの`BasicLM`で`Task`を`Bake`にする |
| 判断モデル | メニュー`CafeNPC/Decider/0 Bake Decider (Play Mode)` |
| 声のモデル | メニュー`CafeNPC/Voice/1 Bake Language Model (Play Mode)` |
| コーデック | メニュー`CafeNPC/Voice/2 Bake Decoder (Play Mode)` |

声のモデルのメニューは、`ShaderGPT/ImportModel`が作った`MioTTS-0.4B-Q4_0.basiclm.prefab`を読む。先に取り込みを済ませておく。

### 4. シーンに置く

ここは自動化していない。置くものは、`CafeNpcBrain`、`CafeDecider`、`CafeNpcController`と、`Runtime/WorldLLM`のワーカーの選出と入力欄、それに席と品物。品物や席の登録はワールドごとに違うので、`CafeNpcBrain.cs`と`CafeNpcController.cs`の公開フィールドを見ながら自分でつなぐことになる。`Prompts/`の2ファイルの中身は、`CafeNpcBrain`の`orderPrompt`と`replyPrompt`に入れる。

声まわりだけは、メニュー`CafeNPC/Voice/3 Install Voice`が配置と参照の設定をする。Play Modeの外で実行する。前提は次の3つ。

- シーンに`CafeNpcBrain`があり、`actor`、`worker`、`pipeline`がつないである
- `WorldLLM`という名前のGameObjectがある
- `CafeVoiceGenerator`と`CafeVoicePlayer`のUdonSharpのプログラムアセットができている

## 構成

```
Runtime/
  CafeNPC/     NPC本体
  WorldLLM/    ワーカーの選出、入力欄、出力フィルタ、同期
  ShaderGPT/   ShaderGPTに上書きするファイル
Editor/        モデルのBakeと、声まわりの配置
Prompts/       注文の読み取りと返事のプロンプト
Tools/         モデルの変換と語彙の絞り込み
LICENSES/      ライセンスの全文
```

`Runtime/CafeNPC`は`CafeNpcBrain.cs`から読むとよい。接客の流れはこのファイルにまとまっている。

| ファイル | 役割 |
|---|---|
| `CafeNpcBrain.cs` | 接客の流れ。用件の分類、注文の読み取り、行動の選択、返事。順番待ちと伝票も持つ |
| `CafeDecider.cs` | 判断モデルに選択肢から1つ選ばせる |
| `CafeNpcController.cs` | NPCの動き。歩行、運搬、表情、口の動き、身ぶり、吹き出し。セリフと音声トークンを全員に同期する |
| `CafeVoiceGenerator.cs` | セリフを音声トークンにする。ワーカーのPCだけで動く |
| `CafeVoicePlayer.cs` | 音声トークンを波形に戻して鳴らす。参加者それぞれのPCで動く |
| `CafeSeat.cs` | だれがどの席にいるかの管理と、座る高さの調整 |
| `CafeSeatSwitch.cs` | 席に座れるかどうかを切り替える壁のスイッチ |
| `CafeVolume.cs` | NPCの声とBGMの音量スライダー。同期しない |
| `CafeEdible.cs` | 届いた品物を食べる、飲む |
| `CafeBell.cs` | 席のベル。NPCを呼ぶ |

| ファイル | 役割 |
|---|---|
| `LlmPipeline.cs` | 会話モデルの呼び出し |
| `LlmChatUI.cs` | 入力欄と履歴の表示 |
| `LlmOutputFilter.cs` | 出力フィルタ |
| `Net/LlmWorkerController.cs` | ワーカーの選出と、入力の受け付け |
| `Net/LlmSelfTest.cs` | ワーカーになる前の、GPUの自己テスト |
| `Net/LlmHeartbeat.cs` | ワーカーが動いているかの通知 |
| `Net/LlmHistory.cs` | 履歴の同期 |
| `Net/LlmResponseStream.cs` | 生成中の返事の同期 |
| `Net/LlmBandwidth.cs` | 同期の送信量の配分 |

## 制限

- NPCは用件の判断も返事もまちがえることがある。出力フィルタはNGワードの部分一致だけで、不適切な返事を防ぎきれない
- 相手にできる客は1人ずつ。ほかの客は順番待ちになる
- 日本語以外の文章は、正しく読み取れないことが多い
- 語彙から長い英単語のトークンを削ってあるので、NPCは英語の文をうまく書けず、英単語の読み上げも不正確になる
- 語彙を絞っても、モデルだけで1.2GB近くある

## ライセンス

このリポジトリのコードはMIT License。`Runtime/ShaderGPT`の4ファイルはShaderGPTのコードを元にしていて、ShaderGPTもMIT License。モデルのライセンスはモデルごとに違う。

`ATTRIBUTION.txt`は公開しているワールドのクレジットをそのまま載せている。このリポジトリに含まないアバターやフォント、BGMの記載もある。ライセンスの全文は`LICENSES/`にある。
