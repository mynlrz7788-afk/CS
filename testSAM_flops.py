import torch
import time
from fvcore.nn import FlopCountAnalysis

# 直接导入模型分配器
from models import get_model

def test_single_model(model_config, device, img_size=1024):
    """测试单个模型的性能"""
    print(f"\n🚀 正在加载并评估模型: [{model_config['name']}] ...")
    
    try:
        # 1. 神奇的一步：直接用字典实例化模型！不需要再手工 import 各种文件了
        model = get_model(model_config).to(device)
        model.eval()

        dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

        # 2. 算参数量
        params = sum(p.numel() for p in model.parameters())

        # 3. 用 fvcore 算 FLOPs
        with torch.no_grad():
            flops_analyzer = FlopCountAnalysis(model, dummy_input)
            flops_analyzer.unsupported_ops_warnings(False)
            flops_analyzer.uncalled_modules_warnings(False)
            
            macs = flops_analyzer.total()
            g_flops = (macs * 2) / 1e9  # 换算成 GFLOPs

        # 4. 测纯净的推理速度 (FPS)
        with torch.no_grad():
            for _ in range(10):  # 预热
                _ = model(dummy_input)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(30):  # 正式测 30 次取平均
                _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()
            fps = 30.0 / (end_time - start_time)

        # 打印这份华丽的成绩单
        print(f"✅ [{model_config['name']}] ")
        print(f"   🔹 参数量 (Params): {params / 1e6:>6.2f} M")
        print(f"   🔹 计算量 (GFLOPs): {g_flops:>6.2f} G")
        print(f"   🔹 推理速度 (FPS):  {fps:>6.2f} frames/s")
        print("-" * 55)

    except Exception as e:
        print(f"❌ 测试 [{model_config['name']}] 时失败，报错信息: {e}")
        print("-" * 55)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =====================================================================
    # 🌟 进阶核心玩法：你的模型清单
    # 就像点菜一样，把你想测的模型名字全写在这个列表里！
    # 这里的字典，就是模拟你在 json 配置文件里 "model": {...} 下面的内容
    # =====================================================================
    model_configs = [
        # 1. 测试经典的 UNet (注意它的参数名是 n_classes)
        {'name': 'UNet', 'n_classes': 1},
        
        # 2. 测试 DeepLabV3+
        {'name': 'DeepLabV3Plus', 'num_classes': 1},
        
        # 3. 测试 DLinkNet
        {'name': 'DLinkNet', 'num_classes': 1},
        
        # 4. 测试你的 AFDANet (假设你叫这个名字，请根据实际 __init__.py 里的名字修改)
        {'name': 'AFDANet', 'num_classes': 1},
        
        # 5. 测试最新加入的 SAM2-UNet (算FLOPs不需要加载真实权重，传个None骗过它就行)
        {'name': 'SAM2UNet', 'num_classes': 1, 'hiera_path': None}
    ]

    print("=" * 55)
    print("=" * 55)

    # 循环遍历菜单，一键全自动测评！
    for config in model_configs:
        test_single_model(config, device, img_size=1024)

    

if __name__ == '__main__':
    main()