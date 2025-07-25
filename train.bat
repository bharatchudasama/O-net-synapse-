@echo off
setlocal

echo.
echo =================================
echo 1. Generating Test File List
echo =================================
echo.

SET TEST_DATA_DIR=D:\Bharat\O-Net-main\data\Synapse\test_vol_h5
SET LIST_FILE=D:\Bharat\O-Net-main\lists\lists_Synapse\test_vol.txt

python prepare_test_list.py --data_dir "%TEST_DATA_DIR%" --output_file "%LIST_FILE%"

REM Check if the Python script ran successfully. If not, stop.
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Failed to generate the test list. Aborting test.
    GOTO END
)

echo.
echo =================================
echo 2. Initializing Test Configuration
echo =================================
echo.

REM --- Use the same configuration as train.bat ---
IF DEFINED epoch_time (
    SET EPOCH_TIME=%epoch_time%
) ELSE (
    SET EPOCH_TIME=300
)

IF DEFINED img_size (
    SET IMG_SIZE=%img_size%
) ELSE (
    SET IMG_SIZE=224
)

IF DEFINED cfg (
    SET CFG=%cfg%
) ELSE (
    SET CFG=configs/swin_tiny_patch4_window7_224_lite.yaml
)

REM --- Model and Data Directories (using absolute paths for reliability) ---
SET MODEL_OUT_DIR=D:\Bharat\O-Net-main\model_out
SET LIST_DIR=D:\Bharat\O-Net-main\lists\lists_Synapse

REM --- Determine the final epoch to test ---
SET /a TEST_EPOCH=%EPOCH_TIME% - 1

echo Configuration Loaded:
echo   - Test Epoch: %TEST_EPOCH%
echo   - Model Directory: %MODEL_OUT_DIR%
echo   - Test Data: %TEST_DATA_DIR%
echo   - List Directory: %LIST_DIR%
echo.

REM --- Construct paths to the final checkpoints ---
SET "CKPT_CNN=%MODEL_OUT_DIR%\CNN_D_model\epoch_%TEST_EPOCH%.pth"
SET "CKPT_SWIN=%MODEL_OUT_DIR%\Swin_D_model\epoch_%TEST_EPOCH%.pth"

REM --- Verify that the checkpoint files exist before testing ---
IF NOT EXIST "%CKPT_CNN%" (
    echo ❌ Error: CNN checkpoint not found!
    echo    Expected at: %CKPT_CNN%
    GOTO END
)
IF NOT EXIST "%CKPT_SWIN%" (
    echo ❌ Error: Swin checkpoint not found!
    echo    Expected at: %CKPT_SWIN%
    GOTO END
)

echo.
echo =================================
echo 3. Starting Model Test on Epoch %TEST_EPOCH%
echo =================================
echo.

REM --- Run the test script using the correct arguments ---
python test_o_net.py ^
    --output_dir %MODEL_OUT_DIR% ^
    --dataset Synapse ^
    --cfg %CFG% ^
    --volume_path %TEST_DATA_DIR% ^
    --list_dir %LIST_DIR% ^
    --epochs_numbers %TEST_EPOCH% ^
    --img_size %IMG_SIZE% ^
    --is_savenii

echo.
echo ✅ Test for epoch %TEST_EPOCH% complete.

:END
pause
