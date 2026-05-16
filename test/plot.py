import numpy as np
import plt


def plot_seismic_data(
        data: np.ndarray,
        x_range: tuple,  # 仅控制X轴刻度数值 (起始道数, 结束道数)，不裁剪数据
        y_range: tuple,  # 仅控制Y轴刻度数值 (起始时间, 结束时间)，不裁剪数据
        save_path: str = "seismic.png",
        title: str = "",
        cmap: str = "seismic",  # 地震专用配色，可选 gray / bwr
        dpi: int = 300  # 高清保存
):
    """
    专业地震数据绘图函数（刻度独立控制，数据完整显示）
    Args:
        data: 二维地震数据 (Time, Trace) → 形状 (H, W)
        vmin: 色标最小值
        vmax: 色标最大值
        x_range: X轴刻度显示范围 (trace_start, trace_end)，数据完整显示
        y_range: Y轴刻度显示范围 (time_start, time_end)，数据完整显示
        save_path: 图片保存路径
        title: 图像标题
        cmap: 配色方案，默认 seismic（地震专业色）
        dpi: 保存分辨率
    """
    # 创建画布
    plt.figure(figsize=(8, 6))
    H, W = data.shape  # 获取数据真实尺寸 (Time, Trace)

    vmin = -np.percentile(np.abs(data),99)
    vmax = np.percentile(np.abs(data),99)
    # 绘制地震图像：完整显示全部数据，时间轴向下
    im = plt.imshow(
        data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        aspect='auto'
    )

    # ====================== 核心修改 ======================
    # 1. X轴：仅设置刻度数值，不裁剪数据（Trace Number）
    x_ticks_pos = np.linspace(0, W-1, 5)  # 固定5个刻度点（可调整）
    x_ticks_label = np.linspace(x_range[0], x_range[1], 5).astype(int)
    plt.xticks(x_ticks_pos, x_ticks_label)

    # 2. Y轴：仅设置刻度数值，不裁剪数据（Time ms）
    y_ticks_pos = np.linspace(0, H-1, 5)
    y_ticks_label = np.linspace(y_range[0], y_range[1], 5).astype(int)
    plt.yticks(y_ticks_pos, y_ticks_label)

    # 设置坐标轴标题
    plt.xlabel("Trace Number", fontsize=12, fontweight='bold')
    plt.ylabel("Time (ms)", fontsize=12, fontweight='bold')

    # 设置标题
    plt.title(title, fontsize=14, fontweight='bold', pad=10)


    # 紧凑布局
    plt.tight_layout()

    # 保存高清图片
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()  # 关闭画布，释放内存

    print(f" 地震图像已保存至：{save_path}")

plot_seismic_data(np.load("../example_data/example_denoised.npy"),(0,480),(0,4000),"../example_data/denoised.png",cmap="seismic",dpi=300)