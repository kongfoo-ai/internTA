#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   WECHAT_APPID=xxx WECHAT_APPSECRET=yyy upload-wechat.sh AUDIO_FILE_PATH
# 
# This script:
#   1. Calls getStableAccessToken to obtain an access_token:
#      https://developers.weixin.qq.com/doc/subscription/api/base/api_getstableaccesstoken.html
#   2. Uses that token to upload a PERMANENT voice media file:
#      https://developers.weixin.qq.com/doc/subscription/api/material/management/api_add_material.html

WECHAT_API_BASE="https://api.weixin.qq.com"

error() {
  echo "Error: $*" >&2
  exit 1
}

print_usage() {
  cat >&2 <<EOF
Usage:
  WECHAT_APPID=xxx WECHAT_APPSECRET=yyy $(basename "$0") AUDIO_FILE_PATH

Notes:
  - AUDIO_FILE_PATH must point to a voice file (<=2MB, AMR/MP3) as required by WeChat.
  - This uploads as a PERMANENT media of type "voice".
EOF
}

main() {
  local appid appsecret audio_file

  if [[ $# -ne 1 ]]; then
    print_usage
    exit 1
  fi

  # Env vars mode only
  appid=${WECHAT_APPID:-}
  appsecret=${WECHAT_APPSECRET:-}
  audio_file=$1

  [[ -z "${appid}" ]] && error "APPID is not set (argument or WECHAT_APPID)."
  [[ -z "${appsecret}" ]] && error "APPSECRET is not set (argument or WECHAT_APPSECRET)."
  [[ -f "${audio_file}" ]] || error "Audio file not found: ${audio_file}"

  if ! command -v curl >/dev/null 2>&1; then
    error "curl is required but not found in PATH."
  fi

  if ! command -v jq >/dev/null 2>&1; then
    error "jq is required to parse JSON (brew install jq)."
  fi

  echo "Requesting stable access_token from WeChat..."

  # getStableAccessToken: POST https://api.weixin.qq.com/cgi-bin/stable_token
  # Payload:
  # {
  #   "grant_type": "client_credential",
  #   "appid": "APPID",
  #   "secret": "APPSECRET"
  # }
  local token_response access_token errcode errmsg

  token_response=$(curl -sS -X POST "${WECHAT_API_BASE}/cgi-bin/stable_token" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg appid "$appid" --arg secret "$appsecret" \
      '{grant_type:"client_credential", appid:$appid, secret:$secret}')" )

  errcode=$(echo "${token_response}" | jq -r 'if has("errcode") then .errcode else 0 end')
  if [[ "${errcode}" != "0" ]]; then
    errmsg=$(echo "${token_response}" | jq -r 'if has("errmsg") then .errmsg else "unknown error" end')
    echo "WeChat stable_token error: errcode=${errcode}, errmsg=${errmsg}" >&2
    echo "Full response: ${token_response}" >&2
    exit 1
  fi

  access_token=$(echo "${token_response}" | jq -r '.access_token // empty')
  if [[ -z "${access_token}" || "${access_token}" == "null" ]]; then
    echo "Failed to parse access_token from response: ${token_response}" >&2
    exit 1
  fi

  echo "Got access_token. Uploading PERMANENT voice media..."

  # upload permanent media:
  # POST https://api.weixin.qq.com/cgi-bin/material/add_material?access_token=ACCESS_TOKEN&type=voice
  local upload_response
  upload_response=$(curl -sS -X POST \
    "${WECHAT_API_BASE}/cgi-bin/material/add_material?access_token=${access_token}&type=voice" \
    -F "media=@${audio_file}")

  errcode=$(echo "${upload_response}" | jq -r 'if has("errcode") and .errcode != 0 then .errcode else 0 end')
  if [[ "${errcode}" != "0" ]]; then
    errmsg=$(echo "${upload_response}" | jq -r 'if has("errmsg") then .errmsg else "unknown error" end')
    echo "WeChat media upload error: errcode=${errcode}, errmsg=${errmsg}" >&2
    echo "Full response: ${upload_response}" >&2
    exit 1
  fi

  echo "Upload succeeded. Response:"
  echo "${upload_response}" | jq .
}

main "$@"

