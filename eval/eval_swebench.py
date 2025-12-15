import subprocess
import os

def main():
    # Set environment variables
    os.environ["DAYTONA_API_KEY"] = "dtn_19b26f08b6a7f295e341c23e64096af54fb2b008e38415c4221f2264f17ff791"
    dataset_path = os.path.expanduser("/home/ubuntu/.cache/huggingface/hub/datasets--DCAgent2--swebench-verified-random-100-folders/snapshots/a2e51e9e0e8029156ed340719eb8cc7ceee3ed1a") 
    processes = []

    for i in range(1, 2):  # otagent1 ... otagent8
        job_name = f"sb_ste30_high{i}"
        print(f"=== Launching job: {job_name} ===")

        cmd = [
            "harbor", "run",
            "--path", dataset_path,
            "--agent", "terminus-2",
            "--model", "hosted_vllm/bespokelabs/Qwen3-8B-ot_step30_high",
            "--n-concurrent", "32",
            "--env", "daytona",
            "--agent-kwarg", "api_base=http://localhost:8000/v1",
            "--agent-kwarg", "key=fake_key",
            "--job-name", job_name,
        ]

        # Start process in parallel (don’t wait here)
        proc = subprocess.Popen(cmd)
        processes.append((job_name, proc))

    # Wait for all jobs to finish
    print("\n=== Waiting for all jobs to complete ===")
    for job_name, proc in processes:
        retcode = proc.wait()
        print(f"Job {job_name} exited with code {retcode}")

if __name__ == "__main__":
    main()
