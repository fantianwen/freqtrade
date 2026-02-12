#!/bin/bash
# ============================================================
# Alex Trading Bot V2 - AWS Deployment Script
# ============================================================
#
# Deploys BTCVolAdjustedV2 alongside V1 (V1 keeps running).
# V2 runs on port 5004, separate DB and logs.
#
# Usage:
#   ./deploy_alex.sh              # Full deploy
#   ./deploy_alex.sh --test       # Test SSH only
#   ./deploy_alex.sh --restart    # Restart V2 service only
#   ./deploy_alex.sh --logs       # Tail V2 live logs
#   ./deploy_alex.sh --status     # Check V2 service status
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
LOCAL_MODELS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../volitality-prediction-tool/project/models" && pwd)"

SSH_CMD="ssh -i $PEM_FILE -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=15"
SCP_CMD="scp -i $PEM_FILE -o StrictHostKeyChecking=no"
RSYNC_CMD="rsync -avz --progress -e \"ssh -i $PEM_FILE -o StrictHostKeyChecking=no -o ServerAliveInterval=30\""

echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Alex Trading Bot V2 - AWS Deployment   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Server:  ${GREEN}$AWS_IP${NC}"
echo -e "  Remote:  $REMOTE_DIR"
echo -e "  Local:   $LOCAL_FT_DIR"
echo -e "  Models:  $LOCAL_MODELS"
echo ""

# ── Test connection ──
if [ "$1" == "--test" ]; then
    echo -e "${YELLOW}Testing SSH connection...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "echo -e '${GREEN}Connection OK${NC}'; uname -a; python3 --version; free -h | head -2; df -h / | tail -1"
    exit 0
fi

# ── Logs ──
if [ "$1" == "--logs" ]; then
    echo -e "${YELLOW}Tailing V2 freqtrade logs...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "tail -f $REMOTE_DIR/logs/freqtrade-btc-v2-live-3.log"
    exit 0
fi

# ── Status ──
if [ "$1" == "--status" ]; then
    echo -e "${YELLOW}Checking V2 service status...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "
        sudo systemctl status alex-trading-v2 --no-pager 2>/dev/null || echo 'V2 service not found'
        echo ''
        echo '--- V2 Recent logs ---'
        tail -20 $REMOTE_DIR/logs/freqtrade-btc-v2-live-3.log 2>/dev/null || echo 'No V2 logs yet'
        echo ''
        echo '--- V1 status ---'
        sudo systemctl status alex-trading --no-pager 2>/dev/null | head -5 || echo 'V1 service not found'
    "
    exit 0
fi

# ── Restart only ──
if [ "$1" == "--restart" ]; then
    echo -e "${YELLOW}Restarting alex-trading-v2 service...${NC}"
    $SSH_CMD "$REMOTE_USER@$AWS_IP" "
        sudo systemctl restart alex-trading-v2
        sleep 3
        sudo systemctl status alex-trading-v2 --no-pager | head -10
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

echo -e "${YELLOW}Step 2/6: Uploading V2 strategy + hyperopt params...${NC}"
$SCP_CMD "$LOCAL_FT_DIR/user_data/strategies/BTCVolAdjustedV2.py" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/strategies/"
$SCP_CMD "$LOCAL_FT_DIR/user_data/strategies/BTCVolAdjustedV2.json" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/strategies/"

echo -e "${YELLOW}Step 3/6: Uploading model files (Feb 11, walk-forward)...${NC}"
$SCP_CMD "$LOCAL_MODELS/regression_model_20260211_230145.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_MODELS/vol_regression_2h_20260211_231704.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_MODELS/range_regression_2h_20260211_231704.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"
$SCP_CMD "$LOCAL_MODELS/vol_classifier_2h_20260211_231704.pkl" \
    "$REMOTE_USER@$AWS_IP:$REMOTE_DIR/user_data/models/"

echo -e "${YELLOW}Step 4/6: Uploading V2 live config...${NC}"
$SCP_CMD "$LOCAL_FT_DIR/user_data/config_btc_vol_adjusted_v2_live.json" \
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

echo -e "${YELLOW}Step 6/6: Installing V2 systemd service (alongside V1)...${NC}"

# Generate the V2 service file
SERVICE_FILE=$(cat << 'SERVICEEOF'
[Unit]
Description=Alex BTC Trading Bot V2 (Freqtrade)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=REMOTE_USER_PLACEHOLDER
WorkingDirectory=REMOTE_DIR_PLACEHOLDER
Environment="PATH=REMOTE_DIR_PLACEHOLDER/venv/bin:/usr/local/bin:/usr/bin:/bin"

ExecStart=REMOTE_DIR_PLACEHOLDER/venv/bin/freqtrade trade \
    --config REMOTE_DIR_PLACEHOLDER/user_data/config_btc_vol_adjusted_v2_live.json \
    --strategy BTCVolAdjustedV2 \
    --strategy-path REMOTE_DIR_PLACEHOLDER/user_data/strategies/ \
    --userdir REMOTE_DIR_PLACEHOLDER/user_data \
    --logfile REMOTE_DIR_PLACEHOLDER/logs/freqtrade-btc-v2-live-3.log

Restart=always
RestartSec=30

StandardOutput=append:REMOTE_DIR_PLACEHOLDER/logs/freqtrade_v2_stdout.log
StandardError=append:REMOTE_DIR_PLACEHOLDER/logs/freqtrade_v2_stderr.log

[Install]
WantedBy=multi-user.target
SERVICEEOF
)

# Replace placeholders
SERVICE_FILE=$(echo "$SERVICE_FILE" | sed "s|REMOTE_USER_PLACEHOLDER|$REMOTE_USER|g" | sed "s|REMOTE_DIR_PLACEHOLDER|$REMOTE_DIR|g")

echo "$SERVICE_FILE" | $SSH_CMD "$REMOTE_USER@$AWS_IP" "sudo tee /etc/systemd/system/alex-trading-v2.service > /dev/null"

$SSH_CMD "$REMOTE_USER@$AWS_IP" "
    sudo systemctl daemon-reload
    sudo systemctl enable alex-trading-v2
    sudo systemctl restart alex-trading-v2
    sleep 5
    echo ''
    echo '--- V2 Service Status ---'
    sudo systemctl status alex-trading-v2 --no-pager | head -15
    echo ''
    echo '--- V1 Service Status ---'
    sudo systemctl status alex-trading --no-pager 2>/dev/null | head -5 || echo 'V1 not running as systemd service'
    echo ''
    echo '--- V2 Recent Logs ---'
    tail -20 $REMOTE_DIR/logs/freqtrade-btc-v2-live-3.log 2>/dev/null || tail -20 $REMOTE_DIR/logs/freqtrade_v2_stdout.log 2>/dev/null || echo 'Waiting for logs...'
"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       V2 Deployment Complete!            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Both bots should now be running:"
echo -e "    ${CYAN}V1:${NC} port 5003, DB: tradesv3-btc-kdj-aws-live-mlmodels.sqlite"
echo -e "    ${CYAN}V2:${NC} port 5004, DB: tradesv3-btc-v2-live-3.sqlite"
echo ""
echo -e "  Commands:"
echo -e "    ${CYAN}./deploy_alex.sh --status${NC}   Check V2 status"
echo -e "    ${CYAN}./deploy_alex.sh --logs${NC}     Tail V2 logs"
echo -e "    ${CYAN}./deploy_alex.sh --restart${NC}  Restart V2 bot"
echo ""
echo -e "  SSH directly:"
echo -e "    ${CYAN}ssh -i $PEM_FILE $REMOTE_USER@$AWS_IP${NC}"
echo ""
echo -e "  API servers:"
echo -e "    V1: ${CYAN}http://$AWS_IP:5003/api/v1/ping${NC}"
echo -e "    V2: ${CYAN}http://$AWS_IP:5004/api/v1/ping${NC}"
echo ""
echo -e "  To stop V1 after V2 is verified:"
echo -e "    ${CYAN}ssh -i $PEM_FILE $REMOTE_USER@$AWS_IP 'sudo systemctl stop alex-trading'${NC}"
echo ""
