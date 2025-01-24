<div align="center">

<img src="./assets/logo.png" width="300em"></img>

## **Open-source Omni-modal Foundation Model Supporting Text, Image, Video, and Audio Inputs as Well as Text and Audio Outputs**

<strong>中文 | 
[English](./README.md)</strong>

<p align="center">
Baichuan-Omni-1.5 <a href="https://huggingface.co/baichuan-inc/Baichuan-Omni-1.5">🤗</a> <a href="xxxx">🤖</a> | Baichuan-Omni-1.5-Base <a href="https://huggingface.co/baichuan-inc/Baichuan-Omni-1.5-Base">🤗</a> | Technical Report <a href="https://huggingface.co/datasets/baichuan-inc/OpenMM-Medical">📖</a>
</p>
<p align="center">
OpenMM-Medical <a href="https://huggingface.co/datasets/baichuan-inc/OpenMM-Medical">🤗</a> | openAudioBench <a href="https://huggingface.co/datasets/baichuan-inc/openAudioBench">🤗</a>
</p>
</div>

**Baichuan-Omni-1.5** is the latest end-to-end trained omni-modal large model that supports comprehensive input modalities (text, image, video, audio) and dual output modalities (text and audio). Built upon the Qwen2.5-7B language model, it can process inputs from various modalities and generate high-quality text and speech outputs in a controllable manner.

- **Baichuan-Omni-1.5-Base**: To promote the development of omni-modal models, we have open-sourced a foundational model trained on high-quality, extensive datasets. This model has not undergone supervised fine-tuning (SFT) for instructions, offering great flexibility and serving as the **best-performing foundational omni-modal model** currently available.

- **Baichuan-Omni-1.5**: Leveraging the robust Baichuan-Omni-1.5-Base, this model undergoes end-to-end training with high-quality omni-modal aligned data. Baichuan-Omni-1.5 achieves text, image, video, and audio understanding capabilities comparable to **GPT-4o-mini**.

## 📖 Table of Contents

- [🏁 Baichuan-Omni-1.5](#baichuan-omni-1.5)
- [🧠 Multi-stage Omni-modal Training Framework](#multi-stage-omni-modal-training-framework)
- [📊 Performance Evaluation](#performance-evaluation)
- [🍰 Example Use Cases](#example-use-cases)
- [🚀 Local WebUI Demo](#local-webui-demo)
  - [Image Demo](#image-demo)
  - [Video Demo](#video-demo)
  - [Audio Demo](#audio-demo)
- [⚙️ Fine-tuning](#fine-tuning)
- [📣 Acknowledgments](#acknowledgments)
- [⚠️ Disclaimer](#disclaimer)
- [📜 License](#license)
- [📜 Citation](#citation)

## Baichuan-Omni-1.5

Baichuan-Omni-1.5 represents the latest and most advanced model in the Baichuan-omni series, trained and inferred through an end-to-end approach. Compared to its predecessor, Baichuan-omni, this model demonstrates significant improvements in text/image/audio/video understanding and text/audio generation, alongside new functionalities such as controllable real-time voice dialogue and omni-modal real-time interaction. Key features of Baichuan-Omni-1.5 include:

- **Omni-modal Understanding and Interaction Capabilities**: Accepts images, videos, texts, and audios as inputs, generating high-quality text and speech outputs. It can handle continuous video and audio streams, enabling **real-time voice interactions** with users. In comprehensive benchmarks like OminiBench, Baichuan-Omni-1.5 achieves top-tier performance within the open-source community, surpassing **GPT-4o-mini**.

- **Excellent Visual Capabilities**: On the OpenCompass benchmark suite, Baichuan-Omni-1.5 scores an average of 73.3 across ten visual evaluation sets. Within the 7B parameter range, it outperforms **GPT-4o-mini**, Gemini 1.5 Pro, and Claude 3.5 Sonnet in single-image understanding. Its video comprehension also exceeds that of **GPT-4V**, Claude 3.5 Sonnet, and other open-source omni-modal models.

- **Outstanding Speech Capabilities**: Supports **high-quality controllable bilingual (Chinese and English) real-time conversations**. Baichuan-Omni-1.5 excels in speech understanding tasks (e.g., ASR and STT), surpassing **GPT-4o-realtime**, and demonstrates the highest speech generation performance among open-source models in semantic and acoustic evaluations. Additional capabilities include emotion/speed/style control, voice cloning, and role-playing.

- **Leading Medical Image Understanding**: Achieves state-of-the-art performance on GMAI-MMBench and OpenMM-Medical. Specifically, on OpenMM-Medical, Baichuan-Omni-1.5 scores 83.8% using a 7B LLM, surpassing Qwen2-VL-72B's score of 80.7%.

- **Strong Real-world Understanding and Other Features**: Enhances numerous visual understanding capabilities, handling images of arbitrary aspect ratios up to 1.8 million pixels (e.g., 1344x1344). Scores 68.8 on RealWorldQA, outperforming commercial closed-source models and recent open-source omni-modal models. It also ranks first in both English and Chinese subsets of MMBench with scores of 85.6% and 83.6%, respectively.

### Model Architecture

- **End-to-End Omni-modal Architecture**: Trains different modality encoders/decoders through a multi-stage, end-to-end progressive method to fully leverage rich knowledge across modalities, promoting complementary knowledge integration. During the omni-modal pretraining phase, the model is entirely trained using NTP loss.
- **High-Quality Controllable Voice Solutions**: Redesigns the multimodal system prompt to include traditional text prompts and **voice system prompts** for specifying the model's voice characteristics. The model can flexibly control voice styles via textual or vocal samples during inference, supporting advanced capabilities like end-to-end voice cloning and voice creation.

### Multi-stage Omni-modal Training Framework

<div align="center">
<img src="./assets/train-pipeline.png" , width=80%>
</div>

<br>

### Performance Evaluation

<div align="center">
<img src="./assets/performance.png" , width=80%>
</div>

<br>

<details>

<summary>Click here to view the detailed results of pure text understanding ability.</summary>

#### Pure text understanding ability
<div align="center">
  <table style="margin: 0 auto; text-align: center;">
    <thead>
        <tr>
            <th class="tg-c3ow" colspan="6">Comprehensive Tasks</th>
        </tr>
    </thead>
    <tbody>
    <tr>
        <td class="tg-c3ow">Model</td>
        <td class="tg-c3ow">Size</td>
        <td class="tg-c3ow">MMLU (Acc.)</td>
        <td class="tg-c3ow">CMMLU (Acc.)</td>
        <td class="tg-c3ow">AGIEval (Acc.)</td>
        <td class="tg-c3ow">C-Eval (Acc.)</td>
    </tr>
    <tr>
        <td class="tg-c3ow" colspan="6">Proprietary Models</td>
    </tr>
    <tr>
        <td class="tg-c3ow">GPT 4o</td>
        <td class="tg-c3ow">-</td>
        <td class="tg-c3ow">88.0♢</td>
        <td class="tg-c3ow">78.3♢</td>
        <td class="tg-c3ow">62.3♢</td>
        <td class="tg-c3ow">86.0♢</td>
    </tr>
    <tr>
        <td class="tg-c3ow">GPT 4o mini</td>
        <td class="tg-c3ow">-</td>
        <td class="tg-c3ow">82.0</td>
        <td class="tg-c3ow">67.6</td>
        <td class="tg-c3ow">52.2</td>
        <td class="tg-c3ow">63.4</td>
    </tr>
    <tr>
        <td class="tg-c3ow" colspan="6">Open-source Models (Pure text)</td>
    </tr>
    <tr>
        <td class="tg-c3ow">MAP-Neo</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">58.2</td>
        <td class="tg-c3ow">55.1</td>
        <td class="tg-c3ow">33.9</td>
        <td class="tg-c3ow">57.5</td>
    </tr>
    <tr>
        <td class="tg-c3ow">Qwen1.5-Chat</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">61.5</td>
        <td class="tg-c3ow">68.0</td>
        <td class="tg-c3ow">39.3</td>
        <td class="tg-c3ow">68.8</td>
    </tr>
    <tr>
        <td class="tg-c3ow">Llama3-Instruct</td>
        <td class="tg-c3ow">8B</td>
        <td class="tg-c3ow">67.1</td>
        <td class="tg-c3ow">51.7</td>
        <td class="tg-c3ow">38.4</td>
        <td class="tg-c3ow">50.7</td>
    </tr>
    <tr>
        <td class="tg-c3ow">OLMo</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">28.4</td>
        <td class="tg-c3ow">25.6</td>
        <td class="tg-c3ow">19.9</td>
        <td class="tg-c3ow">27.3</td>
    </tr>
    <tr>
        <td class="tg-c3ow" colspan="6">Open-source Models (Omni-modal)</td>
    </tr>
    <tr>
        <td class="tg-c3ow">VITA</td>
        <td class="tg-c3ow">8x7B</td>
        <td class="tg-c3ow">71.0*</td>
        <td class="tg-c3ow">46.6</td>
        <td class="tg-c3ow">46.2*</td>
        <td class="tg-c3ow">56.7*</td>
    </tr>
    <tr>
        <td class="tg-c3ow">VITA-1.5</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">71.0</td>
        <td class="tg-c3ow">75.1</td>
        <td class="tg-c3ow">47.9</td>
        <td class="tg-c3ow">65.6</td>
    </tr>
    <tr>
        <td class="tg-c3ow">Baichuan-Omni</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">65.3</td>
        <td class="tg-c3ow">72.2</td>
        <td class="tg-c3ow">47.7</td>
        <td class="tg-c3ow">68.9</td>
    </tr>
    <tr>
        <td class="tg-c3ow">MiniCPM-o 2.6</td>
        <td class="tg-c3ow">7B</td>
        <td class="tg-c3ow">65.3</td>
        <td class="tg-c3ow">63.3</td>
        <td class="tg-c3ow">50.9</td>
        <td class="tg-c3ow">61.5</td>
    </tr>
    <tr>
        <td class="tg-c3ow">Baichuan-Omni-1.5</td>
        <td class="tg-c3ow"></td>
        <td class="tg-c3ow"></td>
        <td class="tg-c3ow"></td>
        <td class="tg-c3ow"></td>
        <td class="tg-c3ow"></td>
    </tr>
    </tbody>
   </table>
</div>

</details>

<details>

<summary>Click here to view detailed evaluation results of image understanding ability.</summary>

#### Image understanding ability

<div align="center">
  <table style="margin: 0 auto; text-align: center;">
    <thead>
      <tr>
         <th class="tg-c3ow" colspan="9">Multi-choice &amp; Yes-or-No Question</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Model</td>
        <td>Size</td>
        <td>MMBench-EN (Acc.)</td>
        <td>MMbench-CN (Acc.)</td>
        <td>M3GIA (Acc.)</td>
        <td>SEED-IMG (Acc.)</td>
        <td>MME (Score)</td>
        <td>MMMU-val (Acc.)</td>
        <td>HallusionBench (Acc.)</td>
      </tr>
      <tr>
        <td colspan="9">Proprietary Models</td>
      </tr>
      <tr>
        <td>GPT-4o</td>
        <td>-</td>
        <td>83.4♢</td>
        <td>82.1♢</td>
        <td>59.8♢</td>
        <td>-</td>
        <td>2328.7♢</td>
        <td>69.1♢</td>
        <td>55.0♢</td>
      </tr>
      <tr>
        <td>GPT-4o-mini</td>
        <td>-</td>
        <td>77.7</td>
        <td>76.9</td>
        <td>-</td>
        <td>72.3</td>
        <td>2003.4♢</td>
        <td>60.0♢</td>
        <td>46.1♢</td>
      </tr>
      <tr>
        <td colspan="9">Open Source Models (Vision-Language)</td>
      </tr>
      <tr>
        <td>Qwen2-VL-7B</td>
        <td>7B</td>
        <td>86.4</td>
        <td>81.9</td>
        <td>37.3</td>
        <td>76.5</td>
        <td>2326.8∗</td>
        <td>52.7</td>
        <td>50.6∗</td>
      </tr>
      <tr>
        <td>MiniCPM-Llama3-V 2.5</td>
        <td>8B</td>
        <td>76.7</td>
        <td>73.3</td>
        <td>30.3</td>
        <td>72.4</td>
        <td>2024.6∗</td>
        <td>45.8∗</td>
        <td>42.5</td>
      </tr>
      <tr>
        <td colspan="9">Open Source Models (Omni-modal)</td>
      </tr>
      <tr>
        <td>VITA</td>
        <td>8x7B</td>
        <td>74.7</td>
        <td>71.4</td>
        <td>27.7</td>
        <td>72.6</td>
        <td>2189.1</td>
        <td>45.3</td>
        <td>39.7∗</td>
      </tr>
      <tr>
        <td>VITA-1.5</td>
        <td>7B</td>
        <td>80.8</td>
        <td>80.2</td>
        <td>-</td>
        <td>74.2</td>
        <td>2311.0</td>
        <td>53.1</td>
        <td>44.1</td>
      </tr>
      <tr>
        <td>Baichuan-Omni</td>
        <td>7B</td>
        <td>76.2</td>
        <td>74.9</td>
        <td>34.7</td>
        <td>74.1</td>
        <td>2186.9</td>
        <td>47.3</td>
        <td>47.8</td>
      </tr>
      <tr>
        <td>MiniCPM-o 2.6</td>
        <td>7B</td>
        <td>83.6</td>
        <td>81.8</td>
        <td>-</td>
        <td>75.4</td>
        <td>2372.0*</td>
        <td>51.1</td>
        <td>50.1</td>
      </tr>
      <tr>
        <td>Baichuan-Omni-1.5 </td>
        <td>7B</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

<br>

<div align="center">
  <table style="margin: 0 auto; text-align: center;">
    <thead>
      <tr>
        <th class="tg-c3ow" colspan="9">Visual Question Answering</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Model</td>
        <td>Size</td>
        <td>RealWorldQA (Acc.)</td>
        <td>MMVet (Acc.)</td>
        <td>MathVista-mini (Acc.)</td>
        <td>TextVQA-val (Acc.)</td>
        <td>ChartQA (Acc.)</td>
        <td>OCRBench (Acc.)</td>
      </tr>
      <tr>
        <td colspan="8">Proprietary Models</td>
      </tr>
      <tr>
        <td>GPT-4o</td>
        <td>-</td>
        <td>75.4♢</td>
        <td>69.1♢</td>
        <td>63.8♢</td>
        <td>-</td>
        <td>85.7♢</td>
        <td>73.6♢</td>
      </tr>
      <tr>
        <td>GPT-4o-mini</td>
        <td>-</td>
        <td>67.1♢</td>
        <td>66.9♢</td>
        <td>52.4♢</td>
        <td>66.8</td>
        <td>-</td>
        <td>78.5♢</td>
      </tr>
      <tr>
        <td colspan="8">Open Source Models (Vision-Language)</td>
      </tr>
      <tr>
        <td>Qwen2-VL-7B</td>
        <td>7B</td>
        <td>69.7</td>
        <td>62.0∗</td>
        <td>58.2∗</td>
        <td>84.3∗</td>
        <td>83.0∗</td>
        <td>84.5∗</td>
      </tr>
      <tr>
        <td>MiniCPM-Llama3-V 2.5</td>
        <td>8B</td>
        <td>63.5</td>
        <td>52.0</td>
        <td>54.3∗</td>
        <td>76.6</td>
        <td>72.0</td>
        <td>72.5</td>
      </tr>
      <tr>
        <td colspan="8">Open Source Models (Omni-modal)</td>
      </tr>
      <tr>
        <td>VITA</td>
        <td>8x7B</td>
        <td>59.0</td>
        <td>41.6∗</td>
        <td>44.9∗</td>
        <td>71.8</td>
        <td>76.6</td>
        <td>68.5∗</td>
      </tr>
      <tr>
        <td>VITA-1.5</td>
        <td>7B</td>
        <td>66.8</td>
        <td>51.1∗</td>
        <td>66.2∗</td>
        <td>74.2</td>
        <td>79.6</td>
        <td>75.2∗</td>
      </tr>
      <tr>
        <td>Baichuan-Omni</td>
        <td>7B</td>
        <td>62.6</td>
        <td>65.4</td>
        <td>51.9</td>
        <td>74.3</td>
        <td>79.6</td>
        <td>70.0</td>
      </tr>
      <tr>
        <td>MiniCPM-o 2.6</td>
        <td>7B</td>
        <td>67.7</td>
        <td>65.5</td>
        <td>71.9∗</td>
        <td>80.1</td>
        <td>86.9∗</td>
        <td>89.7∗</td>
      </tr>
       <tr>
        <td>Baichuan-Omni-1.5 </td>
        <td>7B</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </tbody>
  </table>
</div>

</details>

<details>

<summary>Click here to view detailed evaluation results of video understanding ability.</summary>

#### Video understanding ability
<div style="text-align: center;">
  <table style="margin: 0 auto; border-collapse: collapse; text-align: center;">
    <thead>
      <tr>
        <th colspan="7">General VQA&nbsp;&nbsp;&nbsp;</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Model</td>
        <td>Size</td>
        <td># Frames</td>
        <td>MVBench (Acc.)</td>
        <td>Egoschema (Acc.)</td>
        <td>VideoMME (Acc.)</td>
        <td>Perception-Test (Acc.)</td>
      </tr>
      <tr>
        <td colspan="7">Proprietary Models</td>
      </tr>
      <tr>
        <td>Gemini 1.5 Pro</td>
        <td>-</td>
        <td>-</td>
        <td>81.3♢</td>
        <td>63.2*</td>
        <td>75.0♢</td>
        <td>-</td>
      </tr>
      <tr>
        <td>GPT 4o mini</td>
        <td>-</td>
        <td>-</td>
        <td>55.2</td>
        <td>58.5</td>
        <td>65.2</td>
        <td>48.2</td>
      </tr>
      <tr>
        <td>GPT 4o</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>77.2*</td>
        <td>71.9♢</td>
        <td>-</td>
      </tr>
      <tr>
        <td>GPT 4V</td>
        <td>-</td>
        <td>-</td>
        <td>43.7♢</td>
        <td>55.6*</td>
        <td>59.9♢</td>
        <td>-</td>
      </tr>
      <tr>
        <td colspan="7">Open-source Models (Vision-language)</td>
      </tr>
      <tr>
        <td>Qwen2-VL-7B</td>
        <td>7B</td>
        <td>2 fps (max 768)</td>
        <td>67.0* | 64.4</td>
        <td>66.7* | 66.6</td>
        <td>63.3* | 59.0</td>
        <td>62.3* | 60.3</td>
      </tr>
      <tr>
        <td>AnyGPT</td>
        <td>8B</td>
        <td>48</td>
        <td>33.2</td>
        <td>32.1</td>
        <td>29.8</td>
        <td>29.1</td>
      </tr>
      <tr>
        <td>VideoLLaMA 2</td>
        <td>7B</td>
        <td>16</td>
        <td>54.6*</td>
        <td>51.7*</td>
        <td>46.6*</td>
        <td>51.4*</td>
      </tr>
      <tr>
        <td>VideoChat2</td>
        <td>7B</td>
        <td>16</td>
        <td>51.1*</td>
        <td>42.1♢</td>
        <td>33.7♢</td>
        <td>47.3♢</td>
      </tr>
      <tr>
        <td>LLaVA-NeXT-Video</td>
        <td>7B</td>
        <td>32</td>
        <td>46.5♢</td>
        <td>43.9♢</td>
        <td>33.7♢</td>
        <td>48.8♢</td>
      </tr>
      <tr>
        <td>Video-LLaVA</td>
        <td>7B</td>
        <td>8</td>
        <td>41.0♢</td>
        <td>38.4♢</td>
        <td>39.9♢</td>
        <td>44.3♢</td>
      </tr>
      <tr>
        <td colspan="7">Open-source Models (Omni-modal)</td>
      </tr>
      <tr>
        <td>VITA</td>
        <td>8x7B</td>
        <td>1 fps (max 32)</td>
        <td>53.4</td>
        <td>53.9</td>
        <td>56.1</td>
        <td>56.2</td>
      </tr>
      <tr>
        <td>VITA-1.5</td>
        <td>7B</td>
        <td>1 fps (max 32)</td>
        <td>55.5</td>
        <td>54.7</td>
        <td>58.6</td>
        <td>57.6</td>
      </tr>
      <tr>
        <td>Baichuan-Omni</td>
        <td>7B</td>
        <td>1 fps (max 48)</td>
        <td>60.9</td>
        <td>58.8</td>
        <td>58.2</td>
        <td>56.8</td>
      </tr>
      <tr>
        <td>MiniCPM-o 2.6</td>
        <td>7B</td>
        <td>1 fps (max 64)</td>
        <td>58.6</td>
        <td>50.7</td>
        <td>66.7</td>
        <td>66.6</td>
      </tr>
      <tr>
        <td>Baichuan-Omini-1.5</td>
        <td>7B</td>
        <td>1 fps (max 48)</td>
        <td> <strong>63.7 </td>
        <td> <strong>62.4 </td>
        <td> <strong>62.6 </td>
        <td> <strong>68.9 </td>
      </tr>
    </tbody>
   </table>
</div>

<br>

<div style="text-align: center;">
  <table style="margin: 0 auto; border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th colspan="7">Open-ended VQA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Model</td>
      <td rowspan="2">Size</td>
      <td rowspan="2"># Frames</td>
      <td colspan="2">ActivityNet-QA</td>
      <td colspan="2">MSVD-QA</td>
    </tr>
    <tr>
      <td>(Acc.)</td>
      <td>(Score)</td>
      <td>(Acc.)</td>
      <td>(Score)</td>
    </tr>
    <tr>
      <td colspan="7">Proprietary Models</td>
    </tr>
    <tr>
      <td>Gemini 1.5 Pro</td>
      <td>-</td>
      <td>-</td>
      <td>56.7*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GPT 4o mini</td>
      <td>-</td>
      <td>1 fps (max 32)</td>
      <td>59.6</td>
      <td>3.0</td>
      <td>75.8</td>
      <td>3.7</td>
    </tr>
    <tr>
      <td>GPT 4o</td>
      <td>-</td>
      <td>-</td>
      <td>61.9*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GPT 4V</td>
      <td>-</td>
      <td>-</td>
      <td>59.5*</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td colspan="7">Open-source Models (Vision-language)</td>
    </tr>
    <tr>
      <td>Qwen2 VL</td>
      <td>7B</td>
      <td>2 fps (max 768)</td>
      <td>17.4</td>
      <td>1.9</td>
      <td>61.1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <td>VideoLLaMA 2</td>
      <td>7B</td>
      <td>16</td>
      <td>50.2*</td>
      <td>3.3*</td>
      <td>70.9*</td>
      <td>3.8*</td>
    </tr>
    <tr>
      <td>VideoChat2</td>
      <td>7B</td>
      <td>16</td>
      <td>49.1*</td>
      <td>3.3*</td>
      <td>70.0*</td>
      <td>3.9*</td>
    </tr>
    <tr>
      <td>LLaVA-NeXT-Video</td>
      <td>7B</td>
      <td>32</td>
      <td>53.5*</td>
      <td>3.2*</td>
      <td>67.4</td>
      <td>3.4</td>
    </tr>
    <tr>
      <td>Video-LLaVA</td>
      <td>7B</td>
      <td>8</td>
      <td>45.3*</td>
      <td>3.3*</td>
      <td>70.7*</td>
      <td>3.9*</td>
    </tr>
    <tr>
      <td colspan="7">Open-source Models (Omni-modal)</td>
    </tr>
    <tr>
      <td>VITA</td>
      <td>8x7B</td>
      <td>1 fps (max 32)</td>
      <td>55.0</td>
      <td>3.5</td>
      <td>63.9</td>
      <td>3.7</td>
    </tr>
    <tr>
      <td>VITA-1.5</td>
      <td>7B</td>
      <td>1 fps (max 32)</td>
      <td>59.6</td>
      <td>3.0</td>
      <td>57.6</td>
      <td>4.5</td>
    </tr>
    <tr>
      <td>Baichuan-Omni</td>
      <td>7B</td>
      <td>1 fps (max 48)</td>
      <td>58.6</td>
      <td>3.3</td>
      <td>72.2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>MiniCPM-o 2.6</td>
      <td>7B</td>
      <td>1 fps (max 64)</td>
      <td>63.0</td>
      <td>3.1</td>
      <td>73.7</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>Baichuan-Omni-1.5</td>
      <td>7B</td>
      <td>1 fps (max 48)</td>
      <td>  62.0</td>
      <td> <strong> 3.6</td>
      <td> <strong> 74.2</td>
      <td> <strong> 3.6</td>
    </tr>
  </tbody>
 </table>
</div>

</details>

<details>

<summary>Click here to view detailed evaluation results of audio understanding and generation ability.</summary>

#### Audio understanding and generation ability

</details>

<details>

<summary>Click here to view the detailed evaluation results of omni-modal understanding ability.</summary>

#### Omni-modal understanding ability

<div style="text-align: center;">
  <table style="margin: 0 auto; border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th colspan="7">Omni-Undesratnding </th>
    </tr>
  <thead>
  <tbody>
        <tr>
        <td>Model</td>
        <td>Size</td>
        <td>Image & Audio</td>
        <td>Image Caption & Audio</td>
        <td>Image & Audio Transcript</td>
        <td>Image Caption & Audio Transcript</td>
        </tr>
    </thead>
    <tr>
      <td colspan="6">Proprietary Models</td>
    </tr>
    <tr>
      <td>GPT4o-mini</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>37.0</td>
      <td>37.7</td>
    </tr>
    <tr>
      <td colspan="6">Open-source Models (Omni-modal)</td>
    </tr>
    <tr>
      <td>VITA-1.0</td>
      <td>7B</td>
      <td>33.1</td>
      <td>31.8</td>
      <td>42.0</td>
      <td>44.2</td>
    </tr>
    <tr>
      <td>VITA-1.5</td>
      <td>7B</td>
      <td>33.4</td>
      <td>29.6</td>
      <td>48.5</td>
      <td>47.2</td>
    </tr>
    <tr>
      <td>Baichuan-Omni</td>
      <td>7B</td>
      <td>32.2</td>
      <td>26.5</td>
      <td>42.6</td>
      <td>44.2</td>
    </tr>
    <tr>
      <td>MiniCPM-o 2.6</td>
      <td>7B</td>
      <td>40.5</td>
      <td>30.8</td>
      <td>53.2</td>
      <td>46.3</td>
    </tr>
    <tr>
      <td>Baichuan-Omni-1.5</td>
      <td>7B</td>
      <td>42.9</td>
      <td>37.7</td>
      <td>47.9</td>
      <td>46.9</td>
    </tr>
  </tbody>
 </table>
</div>

</details>

<details>

<summary>Click here to view detailed evaluation results of medical image understanding ability.</summary>

#### Medical image understanding ability

<div style="text-align: center;">
  <table style="margin: 0 auto; border-collapse: collapse; text-align: center;">
    <thead>
      <tr>
        <th colspan="7">Medical understanding&nbsp;&nbsp;&nbsp;</th>
      </tr>
    </thead>
    <tbody>
        <tr>
        <td>Model</td>
        <td>Size</td>
        <td>GMAI-MMB-VAL</td>
        <td>BC-MED-MQA</td>
        </tr>
    </thead>
    <tr>
      <td colspan="4">Proprietary Models</td>
    </tr>
    <tr>
      <td>GPT4o-mini</td>
      <td>-</td>
      <td>46.7</td>
      <td>67.5</td>
    </tr>
    <tr>
      <td colspan="4">Open-source Models (Omni-modal)</td>
    </tr>
    <tr>
      <td>VITA-1.5</td>
      <td>7B</td>
      <td>36.7</td>
      <td>56.7</td>
    </tr>
    <tr>
      <td>MiniCPM-o 2.6</td>
      <td>8B</td>
      <td>41.5</td>
      <td>71.8</td>
    </tr>
    <tr>
      <td>Baichuan-Omni-1.5</td>
      <td>7B</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
 </table>
</div>

</details>

### Typical Examples
<br>

<div style="display: flex; flex-direction: column; align-items: center;">
  <img src="./assets/pipeline.png" alt="pipeline" style="margin-bottom: 5px;">
  <img src="./assets/math.png" alt="math" style="margin-bottom: 5px;">
  <img src="./assets/fly_bill.png" alt="fly_bill" style="margin-bottom: 5px;">
</div>

### Local WebUI Demo

#### Preparation

##### Creating a Virtual Environment
```bash
conda create -n baichuan_omni python==3.10
conda activate baichuan_omni
pip install -r baichuan_omni_requirements.txt
```
##### Download the model and modify the model path
Modify MODEL_PATH in web_demo/constants.py to the local model path

#### Image Demo

```bash
cd web_demo
python vision_s2s_gradio_demo_cosy_multiturn.py
```

#### Audio Demo

```bash
cd web_demo
python s2s_gradio_demo_cosy_multiturn.py
```

#### Video Demo

```bash
cd web_demo
python video_s2s_gradio_demo_cosy_singleturn.py
```

### Fine-tuning
Coming soon

### Open-source Evaluation Datasets

**OpenMM-Medical**

To comprehensively evaluate the model's multi-modal medical capabilities, we have constructed OpenMM-Medical, which includes data from 42 publicly available medical image datasets such as ACRIMA (retinal images), BioMediTech (microscope images), and CoronaHack (X-rays), totaling 88,996 images.

**OpenAudioBench**

To efficiently assess the model's "IQ" issues, we developed OpenAudioBench, comprising five end-to-end audio understanding sub-datasets: four public benchmarks (Llama Question, WEB QA, TriviaQA, AlpacaEval), and an internally created speech logical reasoning dataset by the Baichuan team, totaling 2,701 entries. This suite reflects the model's comprehensive "IQ" level.

### Acknowledgments

- **Visual Encoder Architecture**: [NaVit](https://arxiv.org/abs/2307.06304v1)
- **Automatic Speech Recognition (ASR) Model**: [Whisper](https://github.com/openai/whisper)
- **Large Language Model (LLM)**: [Qwen2.5 7B](https://arxiv.org/abs/2412.15115)
- **Visual Encoder Weight Initialization**: Based on Qwen2-VL-7B ([Link](https://arxiv.org/abs/2409.12191))
- **Some Code Contributions**: From CosyVoice and Matcha-TTS ([CosyVoice GitHub](https://github.com/FunAudioLLM/CosyVoice), [Matcha-TTS GitHub](https://github.com/shivammehta25/Matcha-TTS/))
- **HiFi-GAN Vocoder Used in CosyVoice 2.0**: ([CosyVoice 2.0](https://funaudiollm.github.io/cosyvoice2/))

### Disclaimer

We strongly urge all users not to employ the Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base models for any activities that may endanger national or social security or engage in illegal activities. Additionally, we request that these models not be used in internet services without proper safety reviews and registrations. We hope all users adhere to these guidelines to ensure technological development proceeds within a regulated and legal framework.

We have made every effort to ensure the compliance of the data used during the training process. However, despite our extensive efforts, due to the complexity of models and data, unforeseen issues may still arise. Therefore, we will not be held responsible for any problems arising from the use of the Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base open-source models, including but not limited to data security issues, public opinion risks, or risks associated with misleading, misuse, dissemination, or improper utilization of the models.

### License

Community use of the Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base models must comply with the Apache 2.0 license and the "Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base Community License Agreement." These models support commercial use. If you plan to use the Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base models or their derivatives for commercial purposes, please confirm that your entity meets the following criteria:
- Your or your affiliated party's daily active user count (DAU) is below 1 million.
- You or your affiliated party are not software service providers or cloud service providers.
- There is no possibility of re-granting the commercial license to third parties without prior approval from Baichuan Inc.

Under these conditions, you need to submit the required application materials for the "Baichuan-Omni-1.5/Baichuan-Omni-1.5-Base Community License Agreement" via email at opensource.contact@baichuan-inc.com. Upon approval, Baichuan Inc. will grant you a non-exclusive, global, non-transferable, non-sublicensable, and revocable commercial license.

### Citation

If you wish to cite our work, please use the following reference:
@article{
}