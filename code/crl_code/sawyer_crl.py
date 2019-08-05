import argparse


ENVIRONMENTS = {
    'reach': {
        'default': SawyerReach
    },
}
AVAILABLE_DOMAINS = set(ENVIRONMENTS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain',
                        type=str,
                        choices=AVAILABLE_DOMAINS,
                        default='ant-cross-maze')
    parser.add_argument('--policy',
                        type=str,
                        choices=('gaussian','gaussian_ptr'),
                        default='gaussian_ptr')
    parser.add_argument('--env', type=str, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args