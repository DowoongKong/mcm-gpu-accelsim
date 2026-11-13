#!/bin/bash
# Writes llm_hypothesis1/STATUS.md: running simulations, capture chains, injection/V-a arms, disk. Run once or in a loop.
H=/home/user/accel-sim-docker/llm_exp; L=$H/llm_hypothesis1/Llama-3.3-70B-Instruct; O=$H/llm_hypothesis1/STATUS.md
{
echo "# 진행 상황 스냅샷 $(date '+%F %T')  (디스크 여유 $(df -BG --output=avail $H | tail -1 | tr -dc 0-9) GB, load $(cut -d' ' -f1 /proc/loadavg))"
echo; echo "## 실행 중인 시뮬레이션"; ps -eo nlwp,etime,args | grep "accel-sim.out -config" | grep -v grep | awk '{print "- " $1 "t, 경과 " $2 ", " $NF}' | sed 's#/accel-sim/llm_exp/llm_hypothesis1/##'; 
echo; echo "## 캡처 체인 (마지막 줄)"; for m in Llama-3.3-70B-Instruct Mixtral-8x7B-v0.1 Qwen3-30B-A3B Qwen3-32B; do f=$H/llm_hypothesis1/$m/chain_v2.log; [ -e $f ] && echo "- $m: $(tail -n 1 $f | cut -c1-150)"; done
D=$L/s1024_b0004_prefill_l0_decode_layer0_v2; [ -d $D/traces ] && echo "- Llama S1024 트레이스: 마지막 커널 $(ls $D/traces 2>/dev/null | grep -o 'kernel-[0-9]*' | cut -d- -f2 | sort -n | tail -1), $(du -sh $D/traces 2>/dev/null | cut -f1)$( [ -e $D/exit_code.txt ] && echo ", capture $(cat $D/exit_code.txt)")"
echo; echo "## arm 상태"; for a in sim_va/k71_ref sim_va/k78_ref sim_inj/base_initft sim_inj/ptwhop0_initft; do x=$L/s256_b0001_decode_layer0_v2/$a; echo "- b1 v2 $a: $(cat $x/exit_code.txt 2>/dev/null || ( [ -e $x/run.log ] && echo "실행 중 ($(grep -c '^thread block =' $x/run.log) CTA 발행)" || echo 미시작))"; done
for a in ref_mcm base_mcm ptwhop0_mcm; do x=$D/sim/$a; echo "- S1024 b4 $a: $(cat $x/exit_code.txt 2>/dev/null || ( [ -e $x/run.log ] && echo 실행중 || echo 대기))"; done
echo; echo "## 디스크 큰 항목"; du -sBG $L/*/traces /home/user/hf_cache/_tmp_* 2>/dev/null | sort -rn | head -6 | sed "s#$H/##"
} > $O.tmp && mv $O.tmp $O
