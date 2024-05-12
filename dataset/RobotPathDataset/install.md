# Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/scleronomic/RobotPathData.git
   ```

2. Install the `wzk` package using `pip`:
   ```shell
   pip install wzk
   ```
   - **Note:** You must have a GitHub key. Ensure your `id_rsa.pub` key is added to your GitHub account.

### Error: Fork Method Not Found

If you encounter an error indicating the "fork" method is not found, here's how to fix it:

1. Open the `mp2.py` file where the error occurs.
2. Change the following line:
   ```python
   multiprocessing.set_start_method("fork", force=True)
   ```
   to:
   ```python
   multiprocessing.set_start_method("spawn", force=True)
   ```

