import gradio as gr
import numpy as np
import matplotlib.pyplot as plt


def fake_embedding(audio):
    """
    占位推理函数：不实际加载 pyannote/embedding 模型，
    仅根据输入音频长度随机生成一段嵌入向量，并给出简单可视化。
    """
    # audio: (sr, data) 形式；这里只用长度构造随机向量
    if audio is None:
        return "请先上传一段语音信号。", None

    sr, data = audio
    length = len(data)
    dim = 256

    # 根据长度设置随机种子，使得同一输入在本演示中可复现
    rng = np.random.default_rng(seed=length % (2**32 - 1))
    embedding = rng.normal(size=(dim,))

    # 可视化：简单绘制嵌入维度上的幅度曲线
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(embedding)
    ax.set_title("伪造的说话人嵌入向量幅度分布")
    ax.set_xlabel("维度 index")
    ax.set_ylabel("幅度")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    text = (
        f"本 Demo 未实际加载 pyannote/embedding 模型，仅生成维度为 {dim} 的随机向量。\n"
        f"真实部署时，可使用 pyannote.audio 中的 Model / Inference 接口加载预训练模型，"
        f"并将其替换本函数中的占位实现。"
    )
    return text, fig


with gr.Blocks(title="pyannote Speaker Embedding WebUI Demo") as demo:
    gr.Markdown(
        """
        # 说话人嵌入可视化 WebUI（演示版）

        本界面模拟基于 pyannote 说话人嵌入模型的交互式前端，仅用于展示交互流程与可视化形式，
        不在本地下载或推理真实模型参数。
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="numpy",
                label="上传或录制一段语音",
            )
            run_btn = gr.Button("计算嵌入并可视化", variant="primary")
        with gr.Column(scale=1):
            info_out = gr.Textbox(
                label="嵌入说明（文本）",
                lines=6,
            )
            fig_out = gr.Plot(label="嵌入向量可视化")

    run_btn.click(fn=fake_embedding, inputs=audio_in, outputs=[info_out, fig_out])

if __name__ == "__main__":
    # 演示环境下启一个本地端口用于截图，禁用队列避免不必要的排队逻辑
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)

