import subprocess

subprocess.run(["python", "reinforce_roboschool.py",
                "--hidden", "16", "16",
                "--n_eps", "5000",
                "--n_iter", "10000",
                "--gamma", "0.99",
                "--sigma_sq", "0.01",
                "--lr", "0.001",
                "--env_id", "RoboschoolAnt-v1"])
