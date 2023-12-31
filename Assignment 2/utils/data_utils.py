import os
import json


def save_logs(
        args,
        train_losses,
        train_accs,
        train_times,
        valid_losses,
        valid_accs,
        valid_times,
        test_loss,
        test_acc,
        test_time
    ):
    log_dir = os.path.join(args.log_dir, args.exp_id)
    os.makedirs(log_dir, exist_ok=True)

    # Log arguments
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Log train, val, and test losses and perplexities
    with open(os.path.join(log_dir, "train_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_losses))
    with open(os.path.join(log_dir, "train_accs.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_accs))
    with open(os.path.join(log_dir, "train_time.txt"), "w") as f:
        f.write("\n".join(str(item) for item in train_times))
    with open(os.path.join(log_dir, "valid_loss.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_losses))
    with open(os.path.join(log_dir, "valid_accs.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_accs))
    with open(os.path.join(log_dir, "valid_time.txt"), "w") as f:
        f.write("\n".join(str(item) for item in valid_times))
    with open(os.path.join(log_dir, "test_loss.txt"), "w") as f:
        f.write(f"{test_loss}\n")
    with open(os.path.join(log_dir, "test_acc.txt"), "w") as f:
        f.write(f"{test_acc}\n")
    with open(os.path.join(log_dir, "test_time.txt"), "w") as f:
        f.write(f"{test_time}\n")
