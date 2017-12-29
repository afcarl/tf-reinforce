import subprocess

subprocess.run(["python", "reinforce.py",
                "--hidden", "64", "64",
                "--n_eps", "2000",
                "--gamma", "0.99",
                "--lr", "0.001",
                "--var", "0.01",
                "--env_id", "RoboschoolAnt-v1"])
