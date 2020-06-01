import os
import glob
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from utils import AverageMeter, nsml_report
from utils.metrics import compute_nme, decode_preds, decode_preds_2
from utils.vis import visualize, visualize_batch
    

def train(data_loader, model, hm_criterion, off_criterion, optimizer, epoch, global_iter, opt, nsml=None, scope=None):
  model.cuda()

  print("Start Training...")
  losses = AverageMeter()
  model.train()

  for i, (img, heatmaps, offsets, _) in enumerate(data_loader):
    img = torch.Tensor(img).float().cuda(non_blocking=True)

    # Prediction
    outs = model(img)

    # Backward and step
    optimizer.zero_grad()

    hm_loss = 0
    off_loss = 0

    for os in range(len(outs)):
      pred_map, pred_x, pred_y = outs[os]

      true_map = torch.Tensor(heatmaps[os]).float().cuda(non_blocking=True)
      true_x, true_y = [torch.Tensor(off).float().cuda(non_blocking=True) for off in offsets[os]]

      # Only Calculate Losses on Positive Pixel if offset is point mode
      if opt.offset_mode == 'point':
        pred_x = pred_x * true_map
        pred_y = pred_y * true_map

      hm_loss += hm_criterion(pred_map, true_map)
      off_loss += off_criterion(pred_x, true_x)
      off_loss += off_criterion(pred_y, true_y)

    loss = opt.alpha * hm_loss + off_loss

    loss.backward()
    optimizer.step()

    # Update Results
    global_iter += 1

    losses.update(loss.item(), img.size(0))
    current_lr = optimizer.param_groups[0]['lr']


    if (i+1) % 10 == 0:
      print('Epoch[{:3d}]-It:[{:3d}/{:3d}][{:3d}] | '
            'Loss {loss.val:.8f}({loss.avg:.8f}) | '
            'Hm Loss {hm_loss:.4f} | Off Loss {off_loss:.4f} | '
            'LR {lr:.6f},'.format(
               epoch, i+1, len(data_loader), global_iter, 
               loss=losses,
               hm_loss=hm_loss, off_loss=off_loss,
               lr=current_lr)
      )

  if nsml is not None:
    summary = dict()
    summary['train__loss'] = losses.avg
    summary['train__lr'] = current_lr
    nsml_report(nsml, epoch, scope, summary)
    
  print(">>> Epoch[{:3d}] Training Loss : {:.8f}".format(epoch, losses.avg))
  return global_iter

def validate(data_loader, model, hm_criterion, off_criterion, epoch, opt):
  print("\nStart Evaluation...")
  model.cuda()
  model.eval()

  losses, nme = AverageMeter(), AverageMeter()

  for i, (img, heatmaps, offsets, meta) in enumerate(data_loader):
    img = torch.Tensor(img).float().cuda(non_blocking=True)

    # Prediction
    with torch.no_grad():
      outs = model(img)

    hm_loss = 0
    off_loss = 0

    for os in range(len(outs)):
        pred_map, pred_x, pred_y = outs[os]

        true_map = torch.Tensor(heatmaps[os]).float().cuda(non_blocking=True)
        true_x, true_y = [torch.Tensor(off).float().cuda(non_blocking=True) for off in offsets[os]]

        # Only Calculate Losses on Positive Pixel if offset is point mode
        if opt.offset_mode == 'point':
          pred_x = pred_x * true_map
          pred_y = pred_y * true_map

        hm_loss += hm_criterion(pred_map, true_map)
        off_loss += off_criterion(pred_x, true_x)
        off_loss += off_criterion(pred_y, true_y)

    loss = opt.alpha * hm_loss + off_loss

    losses.update(loss.item(), img.size(0))

    pred_map, pred_x, pred_y = outs[-1]

    score_map = pred_map.data.cpu()
    
    if opt.offset_dim==2:
      preds = decode_preds(score_map, [pred_x, pred_y], meta['center'], meta['scale'], [opt.in_res//opt.output_stride[-1], opt.in_res//opt.output_stride[-1]], opt.offset_norm)
    elif opt.offset_dim==1:
      preds = decode_preds_2(score_map, [pred_x, pred_y], meta['center'], meta['scale'], [opt.in_res//opt.output_stride[-1], opt.in_res//opt.output_stride[-1]], opt.offset_norm)

    nme_batch = compute_nme(preds, meta)
    nme.update(np.mean(nme_batch), img.size(0))

    if (i+1) % 10 == 0:
      print('Epoch[{:3d}]-It:[{:3d}/{:3d}] | '
            'Loss {loss.val:.8f} ({loss.avg:.8f}) | '
            'NME(interocular) {NME_interocular:.4f}'.format(
               epoch, i+1, len(data_loader),
               loss=losses,
               NME_interocular=nme.avg)
      )
    
  print('>>> [{:3d}] Validation Result : '
        'Loss: {loss:.8f} | '
        'NME_interocular: {nme:.8f}'.format(
          epoch,
          loss=losses.avg,
          nme=nme.avg)
  )

  return losses.avg, nme.avg


def update_best_score(valid_loss, valid_nme, best_nme, best_epoch, net, epoch, opt, clear_dir=True):
  if valid_nme < best_nme:
    best_epoch = epoch
    best_nme = valid_nme
    print('Best Score Updated...')

    if clear_dir:
      for path in glob.glob('%s/*.pth*' % opt.exp):
        os.remove(path)

    model_filename = '%s/epoch_%04d_NME%.4f_loss%.8f.pth' % (opt.exp, epoch, best_nme, valid_loss)

    # Single GPU
    if opt.ngpu == 1:
      torch.save(net.cpu().state_dict(), model_filename)
    # Multi GPU
    else:
      torch.save(net.cpu().module.state_dict(), model_filename)

  print('>>> Current best: NME: %.8f in %3d epoch\n' % (best_nme, best_epoch))
  
  net.cuda() # CPU -> GPU
  return best_nme, best_epoch


def evaluate(dataset_val, net, hm_criterion, off_criterion, optimizer, epoch, opt, best_nme, best_epoch, global_iter, nsml=None, scope=None):
  valid_loss, valid_nme = \
    validate(dataset_val, 
              net, hm_criterion, off_criterion, 
              epoch, opt)

  if opt.vis:
    visualize(dataset_val, net, epoch, opt)

  if nsml is not None:
    summary = dict()
    summary['test__loss'] = valid_loss
    summary['test__nme'] = valid_nme
    nsml_report(nsml, epoch, scope, summary)

  
  best_nme, best_epoch = update_best_score(valid_loss, valid_nme, best_nme, best_epoch, net, epoch, opt, clear_dir=True)
  return best_nme, best_epoch


def inference(opt, dataset_val, net):
  print("Start Inference...")
  net.cuda()
  net.mode = 'inference'

  batch_time = AverageMeter()
  data_time = AverageMeter()

  num_classes = opt.n_landmarks
  predictions = torch.zeros((len(dataset_val.dataset), num_classes, 2))

  net.eval()

  nme_count = 0
  nme_batch_sum = 0
  count_failure_008 = 0
  count_failure_010 = 0

  end = time.time()
  global_iter = 0

  with torch.no_grad():
      for imgs, heatmaps, offsets, meta in tqdm(dataset_val, desc='Inference'):
          data_time.update(time.time() - end)

          imgs = torch.Tensor(imgs).float().cuda(non_blocking=True)

          # Prediction
          with torch.no_grad():
            pred_map, pred_x, pred_y = net(imgs)[0]

          score_map = pred_map.data.cpu()
          res = [opt.in_res//opt.output_stride[-1], opt.in_res//opt.output_stride[-1]]

          if opt.offset_dim==2:
            preds = decode_preds(score_map, [pred_x, pred_y], meta['center'], meta['scale'], res, opt.offset_norm)
          elif opt.offset_dim==1:
            preds = decode_preds_2(score_map, [pred_x, pred_y], meta['center'], meta['scale'], res, opt.offset_norm)


          # NME
          nme_temp = compute_nme(preds, meta)

          failure_008 = (nme_temp > 0.08).sum()
          failure_010 = (nme_temp > 0.10).sum()
          count_failure_008 += failure_008
          count_failure_010 += failure_010

          nme_batch_sum += np.sum(nme_temp)
          nme_count = nme_count + preds.size(0)
          for n in range(score_map.size(0)):
              predictions[meta['index'][n], :, :] = preds[n, :, :]

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()

          if opt.vis:
            save_dir = os.path.join(opt.exp, 'vis', 'test')
            if not os.path.exists(save_dir):
              os.makedirs(save_dir)
  
            global_iter = visualize_batch(imgs, heatmaps[1], offsets[1], res, net, save_dir, opt, global_iter)

  nme = nme_batch_sum / nme_count
  failure_008_rate = count_failure_008 / nme_count
  failure_010_rate = count_failure_010 / nme_count

  msg = 'Test Results time : {:.4f} | nme : {:.4f} | FR(8%) : {:.4f} | ' \
        'FR(10%) : {:.4f}'.format(batch_time.avg, 100.0*nme,
                              failure_008_rate, failure_010_rate)
  print(msg)

  return nme, predictions