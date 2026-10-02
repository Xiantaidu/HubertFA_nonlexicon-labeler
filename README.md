# HubertFA_nonlexicon-labeler

从 [HubertFA](https://github.com/Xiantaidu/HubertFA) 合并导出的 `model.onnx`（ONNX_EXPORT_VERSION 5，encoder + nonlexicon labeler + forced alignment 三部分合成）中，**拆出 nonlexicon labeler（CVNT 分支）为独立 ONNX 模型**，并提供最小运行代码。

该模型对输入音频逐帧分类，检测非歌词音素（呼吸声等）：

| 类别索引 | 类别 | 含义 |
| --- | --- | --- |
| 0 | None | 普通语音 |
| 1 | AP | 呼吸/换气声 |
| 2 | EP | 结尾呼吸声 |

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `extract_nonlexicon.py` | 拆分脚本：从合并的 `model.onnx` 中切出 `cvnt_logits` 子图 |
| `run_nonlexicon.py` | 最小运行代码：对音频文件逐帧打标 |
| `requirements.txt` | 运行依赖 |

> 模型文件（`nonlexicon_labeler.onnx`，约 370M）不随仓库分发，请用 `extract_nonlexicon.py` 从 HubertFA 导出的 `model.onnx` 自行拆出，并保留同一目录下的 `vocab.json` 与 `config.json`。

## 安装

```bash
pip install -r requirements.txt
```

## 拆分模型

使用带 `onnx` 的环境（训练环境，见主仓库 `requirements_onnx.txt`）：

```bash
python extract_nonlexicon.py --model model.onnx --out nonlexicon_labeler.onnx
```

- 输入：`waveform`（float32 音频波形）
- 输出：`cvnt_logits`，形状 `[1, 3, T]`，3 类对应 `None/AP/EP`
- 拆出结果与原合并模型的 `cvnt_logits` 输出逐位一致（max diff = 0.0）

子图保留了原模型里的 Hubert encoder 部分，因为 CVNT 分支只接收 encoder 对齐后的特征（`input_feature`），不直接吃波形——只要输入是 wav，encoder 就是必需的。

## 运行

```bash
python run_nonlexicon.py input.wav
```

输出示例：

```
audio: input.wav (12.34s, 1234 frames)

[AP] 3 interval(s):
     1.250s -    1.510s  (len  0.260s, conf 0.912)
     ...

[EP] 1 interval(s):
    11.980s -   12.210s  (len  0.230s, conf 0.877)
```

可选参数：

- `--threshold 0.5`：判定为该音素的概率阈值
- `--min-frames 10`：区间最小帧数（过滤碎片）
- `--plot out.png`：保存概率曲线图

`run_nonlexicon.py` 从脚本所在目录加载 `nonlexicon_labeler.onnx`、`vocab.json`、`config.json`，请将拆出的模型与这两个配置文件放在同一目录。

## 模型细节

- 输入：`waveform`，形状 `[1, n_samples]`，采样率为 `config.json` 中 `mel_spec_config.sample_rate`（单声道 float32）。
- 输出：`cvnt_logits`，形状 `[1, 3, T]`，对第 1 维 softmax、第 2 维 argmax 即得每帧类别。
- 帧长：`mel_spec_config.hop_size / sample_rate` 秒。
