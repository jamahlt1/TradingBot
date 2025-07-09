## Automated Hourly Build/Test Check

This repository includes a script and cron setup to automatically run your build/test command every hour.

### Setup
1. Edit `run_build_check.sh` and replace the placeholder with your actual build/test command.
2. Install the cron job:
   - Run `crontab .cron` to install the hourly job for your user.
3. Output will be logged to `cron_test.log` in the project root.

---

**Note:** You can also set up git hooks (e.g., pre-commit or pre-push) to run tests before pushing code. Let us know if you want this configured!