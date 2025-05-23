import torch
import torch.nn as nn
import time


###flops###
from thop import profile
from calflops import calculate_flops
from fvcore.nn import FlopCountAnalysis



def cal_FLOPs(model, dataloader):
    for i, batch in enumerate(dataloader):
        ###### thop库计算#############
        macs, params = profile(model.module, inputs=batch) # multi gpus, 注意这里返回的实际是Macs
        flops = (2 * macs) / 1e9 # 以G为单位


        ###### catflops库计算#############
         kwargs = {"support_offset": support_offset, "support_x": support_x, "support_y": support_y, "query_offset": query_offset,
                  "query_x": query_x, "query_y": query_y, "epoch": torch.tensor(5),
                  "support_proposals": support_proposals, "query_proposals": query_proposals} # 指定参数名称和具体值
        
        flops, macs, params = calculate_flops(model, kwargs=kwargs, print_results=False,
                    print_detailed=False, output_as_string=False) # 计算
        

        ###### fvcore库计算#############
        macs = FlopCountAnalysis(model, batch).total() #  FPS



def cal_fps(model, input_data, num_iters=100):
    # 热身迭代，以确保GPU处于稳定状态
    for _ in range(10):
        _ = model(input_data)
    
    torch.cuda.synchronize()  # 确保所有GPU同步

    # 计时
    start_time = time.time()

    # 进行推理
    for _ in range(num_iters):
        _ = model(input_data)
    
    torch.cuda.synchronize()  # 确保所有GPU同步
    end_time = time.time()
    # 计算FPS
    total_time = end_time - start_time
    fps = num_iters * batch_size / total_time  # FPS = total frames / total time

    if args.rank == 0:
        print(f"Total Samples: {total_samples}, Total Time: {total_time:.2f}s")
        print(f"FPS: {fps:.2f}")

    dist.destroy_process_group()

    return fps
      

        