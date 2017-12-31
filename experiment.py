import subprocess

subprocess.run(["python", "reinforce.py",
                "--hidden", "16", "16",
                "--n_eps", "5000",
                "--gamma", "0.99",
                "--lr", "0.001",
                "--env_id", "RoboschoolAnt-v1"])
