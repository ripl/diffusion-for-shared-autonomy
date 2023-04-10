#!/usr/bin/env python3

def set_egl_id():
    # NOTE: CUDA_VISIBLE_DEVICES is set a little late by the node. Thus, putting
    # export EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES does NOT work. Thus I do it here manually.
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES', False) and 'EGL_DEVICE_ID' not in os.environ:
        print('pre EGL_DEVICE_ID', os.environ.get('EGL_DEVICE_ID', 'variable not found'))
        cvd = os.environ['CUDA_VISIBLE_DEVICES']
        if ',' in cvd:
            cvd = cvd.split(',')[0]
        os.environ['EGL_DEVICE_ID'] = cvd
        print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', 'variable not found'))
        print('EGL_DEVICE_ID', os.environ.get('EGL_DEVICE_ID', 'variable not found'))


def report_cuda_error(job_name):
    """Checks if torch.cuda.is_available() otherwise, report the jobname to ~/cuda-error.txt"""
    import os
    import torch
    if not torch.cuda.is_available():
        with open(os.path.expandvars('${HOME}/cuda-error.txt'), 'a') as f:
            f.write(job_name + '\n')
        raise RuntimeError('CUDA is not available!!')

def upload_slurm_logs():
    import wandb, os
    # HACK to send the stdout/stderr log to wandb.
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.out.log', policy='end')
    wandb.save(f'./slurm-{os.getenv("SLURM_JOB_ID")}.error.log', policy='end')


def prepare_launch(job_name):
    set_egl_id()
    report_cuda_error(job_name)
