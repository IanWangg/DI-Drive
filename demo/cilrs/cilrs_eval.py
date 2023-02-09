from easydict import EasyDict
import torch
from functools import partial
import smtplib

from core.envs import SimpleCarlaEnv
from core.policy import CILRSPolicy
from core.eval import CarlaBenchmarkEvaluator
from core.utils.others.tcp_helper import parse_carla_tcp
from ding.utils import set_pkg_seed, deep_merge_dicts
from ding.envs import AsyncSubprocessEnvManager, BaseEnvManager
from demo.cilrs.cilrs_env_wrapper import CILRSEnvWrapper

def sendmail(ss):
    s = smtplib.SMTP()
    s.connect('smtp.qq.com', 25)
    s.login('1021662605', 'tuvlzaqpebxfbbie')
    from email.mime.text import MIMEText

    msg = MIMEText(ss, 'plain', 'utf-8')
    msg['Subject'] = ss
    msg['From'] = '1021662605@qq.com'
    msg['To'] = 'qhzhang@link.cuhk.edu.hk'
    s.sendmail('1021662605@qq.com', ['qhzhang@link.cuhk.edu.hk'], msg.as_string())

cilrs_config = dict(
    exp_name='qihang_aco',
    env=dict(
        env_num=1,
        visualize=dict(type='rgb',
            outputs=['video'],
            show_text=False,
            save_dir='/home/ywang3/video/'),
        simulator=dict(
            town='Town02',
            disable_two_wheels=True,
            verbose=False,
            planner=dict(
                type='behavior',
                resolution=1,
            ),
            obs=(
                dict(
                    name='rgb',
                    type='rgb',
                    size=[320, 180],
                    position=[2.0, 0.0, 1.4],
                    rotation=[0,0,0],
                    fov=100,
                ),
                dict(
                    name='bev',
                    type='rgb',
                    size=[320, 320],
                    position=[0.0, 0.0, 28],
                    rotation=[-90,0,0],
                ),
                dict(
                    name='obs',
                    type='rgb',
                    size=[320*4, 180*4],
                    position=[-5.5, 0.0, 2.8],
                    rotation=[-15,0,0],
                ),
            ),
        ),
        wrapper=dict(),
        col_is_failure=True,
        stuck_is_failure=True,
        # ignore_light=False,
        manager=dict(
            auto_reset=False,
            shared_memory=False,
            context='spawn',
            max_retry=1,
        ),
    ),
    server=[dict(carla_host='localhost', carla_ports=[9002, 9004, 2])],
    policy=dict(
        #ckpt_path='./checkpoints/cilrs_train/0.000101-best_ckpt.pth',
        ckpt_path='/home/ywang3/workplace/qihang_di/DI-drive/demo/cilrs/checkpoints_taco/cilrs_train_aco/0.0001-00030_ckpt.pth',
        model=dict(
            num_branch=4,
            backbone='resnet34',
            pretrained=False,
            bn=True
        ),
        eval=dict(
            evaluator=dict(
                # suite=['FullTown01-v1'],
                suite=['FullTown02-v2'],
                transform_obs=True,
                render=True,
                seed=1
            ),
        )
    ),
)

main_config = EasyDict(cilrs_config)


def wrapped_env(env_cfg, host, port, tm_port=None):
    return CILRSEnvWrapper(SimpleCarlaEnv(env_cfg, host, port))


def main(cfg, seed=0):
    cfg.env.manager = deep_merge_dicts(BaseEnvManager.default_config(), cfg.env.manager)

    tcp_list = parse_carla_tcp(cfg.server)
    env_num = cfg.env.env_num
    assert len(tcp_list) >= env_num, \
        "Carla server not enough! Need {} servers but only found {}.".format(env_num, len(tcp_list))

    carla_env = BaseEnvManager(
        env_fn=[partial(wrapped_env, cfg.env, *tcp_list[i]) for i in range(env_num)],
        cfg=cfg.env.manager,
    )
    carla_env.seed(seed)
    set_pkg_seed(seed)
    cilrs_policy = CILRSPolicy(cfg.policy).eval_mode
    if cfg.policy.ckpt_path is not None:
        print('loading checkpoint')
        state_dict = torch.load(cfg.policy.ckpt_path)
        cilrs_policy.load_state_dict(state_dict)

    # carla_env.enable_save_replay('./video')
    evaluator = CarlaBenchmarkEvaluator(cfg.policy.eval.evaluator, carla_env, cilrs_policy, instance_name=cfg.exp_name)
    success_rate = evaluator.eval()
    evaluator.close()


if __name__ == '__main__':
    # main(main_config)
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp-name', type=str, default='cilrs_train_imagenet')
    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()

    ckpt_dirs = [
        # 'cilrs_train_imagenet', 
        # 'cilrs_train_aco', 
        # 'cilrs_train_random'
        args.exp_name
    ]
    epochs = [
        f'000{i}0' for i in range(1, 11)
    ]

    main_config.server = [dict(carla_host='localhost', carla_ports=[args.port, args.port + 2, 2])]

    for ckpt_dir in ckpt_dirs:
        # /home/ywang3/workplace/DI-drive/demo/cilrs/checkpoints_taco/cilrs_train_aco_carla
        names = [
            f"/home/ywang3/workplace/DI-drive/demo/cilrs/checkpoints_taco/{ckpt_dir}/0.0001-{epoch}_ckpt.pth" for epoch in epochs
        ]
        main_config.exp_name = ckpt_dir
        for name in names:
            main_config.policy.ckpt_path = name
            main_config.policy.eval.evaluator.seed = 0
            main(main_config)
    '''
    # ckpt_names = ['0.0001022', '5.022e-05', '0.0001011', '5.011e-05']
    # ckpt_names = ['0.000101', '0.000103', '0.000104', '0.0001015']
    ckpt_names = ['5.022e-05', '5.011e-05', '5.05e-05', '5.02e-05', ]
    # ckpt_names = ['5.05e-05', '5.02e-05', ]
    # ckpt_names = ['0.0001022', '0.0001011', '0.000105', '0.000102', ]
    # ckpt_names = ['0.0001022', ]
    # ckpt_names = ['0.0001022', '0.0001011', '0.000105', '0.000102']
    # ckpt_names = ['0.0001022', '0.0001011', '0.000105', '0.000102']
    # ckpt_names = ['0.0001', '0.0005', '1e-05', '5e-05', ]
    # ckpt_names = ['0.0001' ]
    methods = ['checkpoints_taco']
    try:
        for method in methods:
            for name in ckpt_names:
                for i in range(3,10):
                # for i in range(6,10):
                    # main_config.policy.ckpt_path = f'./checkpoints_taco/cilrs_train/{name}-{i*10:05d}_ckpt.pth'
                    main_config.policy.ckpt_path = f'./{method}/cilrs_train/{name}-{i*10:05d}_ckpt.pth'
                    print(main_config.policy.ckpt_path)
                    main_config.policy.eval.evaluator.seed = 0
                    main(main_config)
    except:
        sendmail('fail 194')
        import traceback;traceback.print_exc()
    finally:
        sendmail('success 194')
    '''
