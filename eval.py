import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from math import log10
import utils.metrics as mt


def eval_net(net_PRE, net, loader, criterion, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net_PRE.eval()
    net.eval()
    n_val = len(loader)  # number of batch
    tot_FullLoss_PRE = 0
    tot_FullLoss = 0
    tot_ImL2 = 0
    tot_ImL1 = 0
    tot_psnr = 0
    totKspaceL2 = 0
    tot_SSIM = 0.0

    for batch in tqdm(loader):
        masked_Kspace = batch['masked_Kspaces']
        full_Kspace = batch['target_Kspace']
        full_img = batch['target_img']
        sensitivity_map = batch['sensitivity_map']
        masked_Kspace = masked_Kspace.to(device=device, dtype=torch.float32)
        full_Kspace = full_Kspace.to(device=device, dtype=torch.float32)
        full_img = full_img.to(device=device, dtype=torch.float32)
        sensitivity_map = sensitivity_map.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            enh_Kspace, enh_img, ori_img= net_PRE(masked_Kspace,sensitivity_map)
            rec_img, rec_Kspace, F_rec_Kspace = net(enh_Kspace, enh_img)

        FullLoss, ImL2, ImL1, KspaceL2, _, _ = criterion.calc_gen_loss(rec_img, rec_Kspace, full_img, full_Kspace)
        FullLoss_PRE = criterion.calc_PRE_loss(full_img, enh_img, masked_Kspace)
        val_SSIM = mt.ssim(full_img, rec_img)
        tot_FullLoss += FullLoss.item()
        tot_ImL2 += ImL2.item()
        tot_FullLoss_PRE += FullLoss_PRE.item()
        psnr = 10 * log10(full_img.max()**2 / ImL2.item())

        tot_psnr += psnr
        tot_ImL1 += ImL1.item()
        totKspaceL2 += KspaceL2.item()
        tot_SSIM += val_SSIM


    net.train()
    return rec_img, full_img, F_rec_Kspace, tot_FullLoss / n_val, tot_ImL2 / n_val, tot_ImL1 / n_val, \
           totKspaceL2 / n_val, tot_psnr / n_val, tot_SSIM / n_val, tot_FullLoss_PRE / n_val
