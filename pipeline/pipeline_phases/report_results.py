




def print_pid_mi(pid_results,mi_result):
    """This functions takes the pid results and the mutual information results and prints them in a nice format."""
    print("PID Results:")
    for key, value in pid_results.items():
        print(f"  {key}: {value}")
    print("Mutual Information Results:")
    for key, value in mi_result.items():
        print(f"  {key}: {value}")