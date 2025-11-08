from Create_dataset import *
from utils import *
from builder_resnet34_final import NetworkStage2
from options import *
from saver import resume, save_img_single
from tqdm import tqdm
from config import config
from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile
# 暂时屏蔽输出
import sys
import os
# sys.stdout = open(os.devnull, 'w')
def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()
    # define model, optimiser and scheduler
    # print(opts)
    device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
    MPF_model = NetworkStage2(n_classes=opts.class_nb).to(device)
    MPF_model = resume(MPF_model, model_save_path=opts.resume, device=device, is_train=False)

    # define dataset
    test_dataset = FusionData(opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False)

    # Train and evaluate multi-task network
    multi_task_tester(test_loader, MPF_model, device, opts)


def multi_task_tester(test_loader, multi_task_model, device, opts):
    multi_task_model.eval()
    test_bar = tqdm(test_loader)
    ## define save dir
    Fusion_save_dir = os.path.join(opts.result_dir, opts.dataname)
    os.makedirs(Fusion_save_dir, exist_ok=True)
    flop_total = 0
    with torch.no_grad():  # operations inside don't track history
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            
            Seg_pred, out1, out2, out3, out4, out5, out6, fused_img = multi_task_model(img_vi, img_ir)

            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                save_img_single(fused_img[i, ::], fusion_save_name, widths[i], heights[i])
                test_bar.set_description('Image: {} '.format(img_name))
    

if __name__ == '__main__':
    main()
