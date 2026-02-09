#!/bin/bash
# ============================================================
# Alex Trading Bot - AWS Deployment Script
# ============================================================
#
# Usage:
#   ./deploy_alex.sh              # Full deploy
#   ./deploy_alex.sh --test       # Test SSH only
#   ./deploy_alex.sh --restart    # Restart service only
#   ./deploy_alex.sh --logs       # Tail live logs
#   ./deploy_alex.sh --status     # Check service status
#
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Config ──
AWS_IP="54.250.16.16"
PEM_FILE="/Users/fantianwen/Projects/Alex/trading-bot.pem"
REMOTE_USER="ubuntu"
REMOTE_DIR="/home/$REMOTE_USER/alex-trading"
LOCAL_FT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SSH_CMD="ssh -i $PEM_FILE -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=15"
SCP_CMD="scp -i $PEM_FILE -o StrictHostKeyChecking=no"
RSYNC_CMD="rsync -avz --progress -e \"ssh -i $PEM_FILE -o StrictHostKeyChecking=no -o ServerAliveInterval=30\""

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║    Alex Trading Bot - AWS Deployment     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Server:  ${GREEN}$AWS_IP${NC}"
echo -e "  Remote:  $REMOTE_DIR"
echo -e "  Local:   $LOCAL_FT_DIR"
echo ""

# ── Test connection ──
if [ "$1" == "--test" ]; then
    echo -e "${YELLOW}Testing SSH connection...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "echo -e '${GREEN}Connection OK${NC}'; uname -a; python3 --version; free -h | head -2; df -h / | tail -1"
    exit 0
fi

# ── Logs ──
if [ "$1" == "--logs" ]; then
    echo -e "${YELLOW}Tailing freqtrade logs...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "tail -f $REMOTE_DIR/logs/freqtrade.log"
    exit 0
fi

# ── Status ──
if [ "$1" == "--status" ]; then
    echo -e "${YELLOW}Checking service status...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "
        sudo systemctl status alex-trading --no-pager 2>/dev/null || echo 'Service not found'
        echo ''
        echo '--- Recent logs ---'
        tail -20 $REMOTE_DIR/logs/freqtrade.log 2>/dev/null || echo 'No logs yet'
    "
    exit 0
fi

# ── Restart only ──
if [ "$1" == "--restart" ]; then
    echo -e "${YELLOW}Restarting alex-trading service...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "
        sudo systemctl restart alex-trading
        sleep 3
        sudo systemctl status alex-trading --no-pager | head -10
    "
    exit 0
fi

# ═══════════════════════════════════════════
# Full deployment
# ═══════════════════════════════════════════

echo -e "${YELLOW}Step 1/6: Creating remote directories...${NC}"
$SSH_CMD "$REMOTE_USER@$AWS_IP" "
    mkdir -p $REMOTE_DIR/{user_data/strategies,user_data/models,user_data/data,logs}
"

echo -e "${YELLOW}Step 2/6: Uploading strategy file...${NC}"
$SCP_CMD "$LOCAL_FT_DIR/user_data/strategies/BTCVolAdjusted.py" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/strategies/"

echo -e "${YELLOW}Step 3/6: Uploading model files...${NC}"
$SCP_CMD "$LOCAL_FT_DIR/user_data/models/regression_model_20260207_204516.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_FT_DIR/user_data/models/vol_regression_2h_20260207_214300.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_FT_DIR/user_data/models/range_regression_2h_20260207_214300.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_FT_DIR/user_data/models/vol_classifier_2h_20260207_214300.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"

echo -e "${YELLOW}Step 4/6: Uploading config...${NC}"
$SCP_CMD "$LOCAL_FT_DIR/user_data/config_btc_vol_adjusted_live.json" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/"

echo -e "${YELLOW}Step 5/6: Installing freqtrade & dependencies on server...${NC}"
$SSH_CMD "$REMOTE_USER@$AWS_IP" "
    cd $REMOTE_DIR

    # Install freqtrade if not present
    if ! command -v freqtrade &>/dev/null && [ ! -f venv/bin/freqtrade ]; then
        echo 'Installing freqtrade...'
        python3 -m venv venv 2>/dev/null || python3.11 -m venv venv 2>/dev/null || python3.10 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip -q
        pip install freqtrade -q
        pip install scikit-learn pandas numpy -q
        echo 'Freqtrade installed.'
    else
        echo 'Freqtrade already available.'
        if [ -f venv/bin/activate ]; then
            source venv/bin/activate
            pip install --upgrade freqtrade scikit-learn -q
        fi
    fi

    # Verify
    if [ -f venv/bin/freqtrade ]; then
        source venv/bin/activate
        freqtrade --version
    elif command -v freqtrade &>/dev/null; then
        freqtrade --version
    fi
"

echo -e "${YELLOW}Step 6/6: Installing systemd service...${NC}"

# Generate the service file
SERVICE_FILE=$(cat << 'SERVICEEOF'
[Unit]
Description=Alex BTC Trading Bot (Freqtrade)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=REMOTE_USER_PLACEHOLDER
WorkingDirectory=REMOTE_DIR_PLACEHOLDER
Environment="PATH=REMOTE_DIR_PLACEHOLDER/venv/bin:/usr/local/bin:/usr/bin:/bin"

ExecStart=REMOTE_DIR_PLACEHOLDER/venv/bin/freqtrade trade \
    --config REMOTE_DIR_PLACEHOLDER/user_data/config_btc_vol_adjusted_live.json \
    --strategy BTCVolAdjusted \
    --strategy-path REMOTE_DIR_PLACEHOLDER/user_data/strategies/ \
    --userdir REMOTE_DIR_PLACEHOLDER/user_data \
    --logfile REMOTE_DIR_PLACEHOLDER/logs/freqtrade.log

Restart=always
RestartSec=30

StandardOutput=append:REMOTE_DIR_PLACEHOLDER/logs/freqtrade_stdout.log
StandardError=append:REMOTE_DIR_PLACEHOLDER/logs/freqtrade_stderr.log

[Install]
WantedBy=multi-user.target
SERVICEEOF
)

# Replace placeholders
SERVICE_FILE=$(echo "$SERVICE_FILE" | sed "s|REMOTE_USER_PLACEHOLDER|$REMOTE_USER|g" | sed "s|REMOTE_DIR_PLACEHOLDER|$REMOTE_DIR|g")

echo "$SERVICE_FILE" | $SSH_CMD "$REMOTE_USER@$AWS_IP" "sudo tee /etc/systemd/system/alex-trading.service > /dev/null"

$SSH_CMD "$REMOTE_USER@$AWS_IP" "
    sudo systemctl daemon-reload
    sudo systemctl enable alex-trading
    sudo systemctl restart alex-trading
    sleep 5
    echo ''
    echo '--- Service Status ---'
    sudo systemctl status alex-trading --no-pager | head -15
    echo ''
    echo '--- Recent Logs ---'
    tail -20 $REMOTE_DIR/logs/freqtrade.log 2>/dev/null || tail -20 $REMOTE_DIR/logs/freqtrade_stdout.log 2>/dev/null || echo 'Waiting for logs...'
"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Deployment Complete!              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Commands:"
echo -e "    ${CYAN}./deploy_alex.sh --status${NC}   Check status"
echo -e "    ${CYAN}./deploy_alex.sh --logs${NC}     Tail logs"
echo -e "    ${CYAN}./deploy_alex.sh --restart${NC}  Restart bot"
echo ""
echo -e "  SSH directly:"
echo -e "    ${CYAN}ssh -i $PEM_FILE $REMOTE_USER@$AWS_IP${NC}"
echo ""
echo -e "  API server (if enabled):"
echo -e "    ${CYAN}http://$AWS_IP:8083/api/v1/ping${NC}"
echo ""
echo -e "  ${YELLOW}NOTE: Bot is running in DRY-RUN mode.${NC}"
echo -e "  To go live, add your Binance API key/secret to the config"
echo -e "  and set dry_run=false."
echo ""
