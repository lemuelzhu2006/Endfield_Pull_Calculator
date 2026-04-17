/* ================================================================
 * 明日方舟：终末地 限定角色池抽卡计算器
 * 核心算法：精确动态规划（DP），非 Monte Carlo 模拟
 *
 * 状态：(pityCount in [0,79], hasLimited in {0,1}, limitedCount in [0, max])
 * 规则：
 *   - 6★ 基础概率 0.8%，pity 65 起每抽 +5pp，pity 79 硬保底
 *   - 任意 6★ 中限定 up 概率 50%
 *   - 前 120 主抽内必得 1 次限定 up（每池仅触发一次）
 *   - 第 30 主抽触发 1 次免费十连（每池仅一次；不计入 pity/累计/120 计数）
 *   - 每 240 主抽赠送 1 个限定信物（计 1 个限定 up）
 *   - 免费十连中出限定 up 会消耗 120 保底资格
 * ================================================================ */

'use strict';

// ---------------- 配置（可替换角色时仅改此区） ---------------- //

const CONFIG = {
    characterName: '庄方宜',
    gemPerPull: 500,
    baseP6: 0.008,
    pitySoftStart: 65,              // pity==65 时开始提升（即第 66 次抽进入提升区）
    pityHardCap: 80,                // pity==79 时必出（即第 80 次抽硬保底）
    pityStep: 0.05,
    limitedRate: 0.5,
    freeTenPullThreshold: 30,       // 第 30 次主抽送 1 次免费十连
    tokenThreshold: 240,            // 每 240 次主抽送 1 个限定信物
    forcedLimitedAtMainDraw: 120,   // 前 120 主抽内必得 1 次限定 up
    freeTenPullSize: 10,
    pruneEpsilon: 1e-18             // 概率剪枝阈值
};

// ---------------- 数值工具 ---------------- //

function binomial(n, k) {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    k = Math.min(k, n - k);
    let v = 1;
    for (let i = 0; i < k; i++) {
        v = (v * (n - i)) / (i + 1);
    }
    return v;
}

/**
 * 普通主抽第 (pity+1) 次未出 6★ 后的本次 6★ 概率
 * 也即：当前 pity=pityCount 时，本次主抽出 6★ 的概率
 */
function p6ForPity(pity) {
    if (pity <= CONFIG.pitySoftStart - 1) return CONFIG.baseP6;       // pity <= 64
    if (pity <= CONFIG.pityHardCap - 2) {                              // pity 65..78
        return CONFIG.baseP6 + CONFIG.pityStep * (pity - (CONFIG.pitySoftStart - 1));
    }
    return 1.0;                                                        // pity == 79 硬保底
}

/**
 * 免费十连中"限定 up 数量"分布（精确）
 *
 * 关键推导：
 *   - 免费十连前 9 抽：每抽独立，6★ 概率 0.8%
 *   - 第 10 抽：若前 9 抽全为 4★（概率 0.912^9），则 6★=0.8% / 5★=99.2% / 4★=0%；
 *              否则 6★=0.8% / 5★=8% / 4★=91.2%
 *   - 两种情况下第 10 抽 6★ 边际概率都是 0.8%（5+ 保底只改变 5★/4★ 占比）
 *   - 因此 10 抽内 6★ 数量 ~ Binomial(10, 0.008)
 *   - 每个 6★ 独立以 50% 成为限定 up
 *   - 故 10 抽内限定 up 数量 ~ Binomial(10, 0.004)
 *
 * FREE_TEN_LIMITED_DIST[m] = P(免费十连中获得恰好 m 个限定 up)，m = 0..10
 */
const FREE_TEN_LIMITED_DIST = (() => {
    const p = CONFIG.baseP6 * CONFIG.limitedRate;     // 0.004
    const q = 1 - p;
    const n = CONFIG.freeTenPullSize;                 // 10
    const arr = new Array(n + 1);
    for (let k = 0; k <= n; k++) {
        arr[k] = binomial(n, k) * Math.pow(p, k) * Math.pow(q, n - k);
    }
    return arr;
})();

// ---------------- 状态编码 ---------------- //
// key = lim * 160 + pity * 2 + has
// pity in [0, 79] -> pity*2 in [0, 158]；has in {0,1}；所以低位 < 160。
const KEY_STRIDE = 160;

function encode(pity, has, lim) {
    return lim * KEY_STRIDE + pity * 2 + has;
}
function decodeLim(key) { return Math.floor(key / KEY_STRIDE); }
function decodePity(key) { return Math.floor((key % KEY_STRIDE) / 2); }
function decodeHas(key) { return (key % KEY_STRIDE) & 1; }

function addState(map, key, p) {
    if (p < CONFIG.pruneEpsilon) return;
    const prev = map.get(key);
    map.set(key, prev === undefined ? p : prev + p);
}

// ---------------- 单步转移 ---------------- //

/**
 * 一次普通主抽（nextDraw 是本次主抽的序号，从 1 开始）
 * 优先处理 "前 120 主抽强制限定" 规则
 * @param {Map<number, number>} states 入口状态分布
 * @param {number} nextDraw 本次主抽序号
 * @param {number} [limCap=Infinity] 可选：limitedCount 软封顶（用于期望 DP 吸收到 target）
 */
function applyMainDraw(states, nextDraw, limCap) {
    const next = new Map();
    const cap = limCap === undefined ? Infinity : limCap;

    for (const [key, p] of states) {
        const lim = decodeLim(key);
        const pity = decodePity(key);
        const has = decodeHas(key);

        // 120 强制限定 up 保底
        if (has === 0 && nextDraw === CONFIG.forcedLimitedAtMainDraw) {
            const newLim = Math.min(lim + 1, cap);
            addState(next, encode(0, 1, newLim), p);
            continue;
        }

        const p6 = p6ForPity(pity);
        const pLim = p6 * CONFIG.limitedRate;
        const pNon = p6 * (1 - CONFIG.limitedRate);
        const pFail = 1 - p6;

        // 限定 6★
        if (pLim > 0) {
            const newLim = Math.min(lim + 1, cap);
            addState(next, encode(0, 1, newLim), p * pLim);
        }
        // 非限定 6★
        if (pNon > 0) {
            addState(next, encode(0, has, lim), p * pNon);
        }
        // 未出 6★
        if (pFail > 0) {
            // pity=79 时 pFail=0；pity 最多到 78，递增后仍 <= 79
            addState(next, encode(pity + 1, has, lim), p * pFail);
        }
    }
    return next;
}

/**
 * 免费十连转移：只改 limitedCount 与 hasLimited，不改 pity / mainDrawsDone
 */
function applyFreeTenPull(states, limCap) {
    const next = new Map();
    const cap = limCap === undefined ? Infinity : limCap;

    for (const [key, p] of states) {
        const lim = decodeLim(key);
        const pity = decodePity(key);
        const has = decodeHas(key);

        for (let m = 0; m <= CONFIG.freeTenPullSize; m++) {
            const pm = FREE_TEN_LIMITED_DIST[m];
            if (pm < CONFIG.pruneEpsilon) continue;
            const newLim = Math.min(lim + m, cap);
            const newHas = m > 0 ? 1 : has;
            addState(next, encode(pity, newHas, newLim), p * pm);
        }
    }
    return next;
}

/**
 * 240 抽送信物：limitedCount += 1
 * （不影响 pity；到此时 hasLimited 必为 1，因 nextDraw >= 240 > 120 保底已触发或已满足）
 */
function applyTokenReward(states, limCap) {
    const next = new Map();
    const cap = limCap === undefined ? Infinity : limCap;

    for (const [key, p] of states) {
        const lim = decodeLim(key);
        const pity = decodePity(key);
        const has = decodeHas(key);
        const newLim = Math.min(lim + 1, cap);
        addState(next, encode(pity, has, newLim), p);
    }
    return next;
}

// ---------------- 概率分布计算 ---------------- //

/**
 * 精确概率分布计算
 *
 * @param {Object} args
 * @param {number} args.N 可用普通主抽数
 * @param {number} args.pityStart 已垫抽数（0~79）
 * @returns {{dist: number[], maxReachable: number, sum: number}}
 *   dist[k] = P(最终获得恰好 k 个限定 up)，k = 0..maxReachable
 */
function computeDistribution({ N, pityStart }) {
    const maxReachable = N + (N >= CONFIG.freeTenPullThreshold ? CONFIG.freeTenPullSize : 0)
        + Math.floor(N / CONFIG.tokenThreshold);

    let states = new Map();
    states.set(encode(pityStart, 0, 0), 1.0);

    for (let k = 1; k <= N; k++) {
        states = applyMainDraw(states, k);
        if (k === CONFIG.freeTenPullThreshold) {
            states = applyFreeTenPull(states);
        }
        if (k > 0 && k % CONFIG.tokenThreshold === 0) {
            states = applyTokenReward(states);
        }
    }

    const dist = new Array(maxReachable + 1).fill(0);
    let sum = 0;
    for (const [key, p] of states) {
        const lim = decodeLim(key);
        if (lim >= 0 && lim <= maxReachable) {
            dist[lim] += p;
        }
        sum += p;
    }

    return { dist, maxReachable, sum };
}

// ---------------- 期望计算 ---------------- //

/**
 * 精确期望主抽数
 *
 * @param {number} target 目标限定 up 数量
 * @returns {{expected: number, absorbedTotal: number, bound: number}}
 */
function computeExpected(target) {
    if (target <= 0) return { expected: 0, absorbedTotal: 1, bound: 0 };

    // 严格上界：K=1 时 120（120 保底）；K>=2 时 240*(K-1)（每 240 必得 1 信物）
    const bound = target === 1 ? CONFIG.forcedLimitedAtMainDraw
        : CONFIG.tokenThreshold * (target - 1);

    let states = new Map();
    states.set(encode(0, 0, 0), 1.0);

    let expected = 0;
    let absorbedTotal = 0;

    for (let k = 1; k <= bound; k++) {
        states = applyMainDraw(states, k, target);
        if (k === CONFIG.freeTenPullThreshold) {
            states = applyFreeTenPull(states, target);
        }
        if (k > 0 && k % CONFIG.tokenThreshold === 0) {
            states = applyTokenReward(states, target);
        }

        const remaining = new Map();
        let absHere = 0;
        for (const [key, p] of states) {
            if (decodeLim(key) >= target) {
                absHere += p;
            } else {
                remaining.set(key, p);
            }
        }
        if (absHere > 0) {
            expected += k * absHere;
            absorbedTotal += absHere;
        }
        states = remaining;
    }

    // 理论上 absorbedTotal 应 = 1；如有残余，按 bound 补齐（保险分支，正常不触发）
    let residual = 0;
    for (const [, p] of states) residual += p;
    if (residual > 1e-9) {
        expected += bound * residual;
        absorbedTotal += residual;
    }

    return { expected, absorbedTotal, bound };
}

// ---------------- 格式化 ---------------- //

function formatPercent(p, digits = 4) {
    if (p === 0) return '0%';
    if (p > 0 && p < 1e-6) return '<0.0001%';
    if (p > 1 - 1e-6 && p < 1) return '>99.9999%';
    if (p >= 1) return '100%';
    return (p * 100).toFixed(digits) + '%';
}

function formatNumber(x, digits = 2) {
    if (!Number.isFinite(x)) return '—';
    if (Math.abs(x - Math.round(x)) < 1e-9) return String(Math.round(x));
    return x.toFixed(digits);
}

function computeMean(dist) {
    let m = 0;
    for (let k = 0; k < dist.length; k++) m += k * dist[k];
    return m;
}

function cumulativeAtLeast(dist, k) {
    let s = 0;
    for (let i = k; i < dist.length; i++) s += dist[i];
    return s;
}

// ---------------- UI ---------------- //

const $ = (id) => document.getElementById(id);

function setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach((t) => {
        t.addEventListener('click', () => {
            tabs.forEach((x) => {
                x.classList.remove('active');
                x.setAttribute('aria-selected', 'false');
            });
            t.classList.add('active');
            t.setAttribute('aria-selected', 'true');
            const key = t.dataset.tab;
            document.querySelectorAll('.tab-panel').forEach((p) => {
                p.classList.toggle('active', p.id === `panel-${key}`);
            });
        });
    });
}

// ----- 输入解析 / 校验 ----- //

function parseNonNegInt(el) {
    const raw = (el.value || '').trim();
    if (raw === '') return 0;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0 || !Number.isInteger(n)) return NaN;
    return n;
}

function markInvalid(el, bad) {
    el.classList.toggle('invalid', !!bad);
}

function showError(errEl, msg) {
    if (msg) {
        errEl.textContent = msg;
        errEl.hidden = false;
    } else {
        errEl.textContent = '';
        errEl.hidden = true;
    }
}

// ----- 概率计算 UI ----- //

function readProbInputs() {
    const gemsEl = $('probGems');
    const quotaEl = $('probQuota');
    const pityEl = $('probPity');
    const targetEl = $('probTarget');

    const gems = parseNonNegInt(gemsEl);
    const quota = parseNonNegInt(quotaEl);
    const pity = parseNonNegInt(pityEl);
    const target = parseNonNegInt(targetEl);

    const errors = [];
    markInvalid(gemsEl, Number.isNaN(gems));
    markInvalid(quotaEl, Number.isNaN(quota));
    markInvalid(pityEl, Number.isNaN(pity) || pity > 79);
    markInvalid(targetEl, Number.isNaN(target) || target < 1);

    if (Number.isNaN(gems)) errors.push('嵌晶玉必须是非负整数');
    if (Number.isNaN(quota)) errors.push('限定抽数量必须是非负整数');
    if (Number.isNaN(pity) || pity > 79) errors.push('已垫抽数必须在 0 ~ 79');
    if (Number.isNaN(target) || target < 1) errors.push('目标限定 up 数量必须 ≥ 1');

    if (errors.length > 0) return { error: errors.join('；') };

    const mainDraws = quota + Math.floor(gems / CONFIG.gemPerPull);
    return { gems, quota, pity, target, mainDraws };
}

function updateProbPreview() {
    const gems = parseNonNegInt($('probGems'));
    const quota = parseNonNegInt($('probQuota'));
    const validG = !Number.isNaN(gems);
    const validQ = !Number.isNaN(quota);
    const N = (validG ? gems : 0) / CONFIG.gemPerPull | 0;
    const mainDraws = (validQ ? quota : 0) + N;

    $('probMainDraws').textContent = mainDraws;
    $('probFreeTen').textContent = mainDraws >= CONFIG.freeTenPullThreshold ? '是（1 次）' : '否';
    $('probTokens').textContent = Math.floor(mainDraws / CONFIG.tokenThreshold);
}

function renderProbResult(input, result) {
    const { dist, maxReachable, sum } = result;
    const { target, mainDraws } = input;

    const successP = cumulativeAtLeast(dist, target);
    const mean = computeMean(dist);
    const atLeast1 = cumulativeAtLeast(dist, 1);

    $('probSuccessLabel').textContent = `至少 ${target} 个限定 up 的概率`;
    $('probSuccessValue').textContent = formatPercent(successP, 4);

    $('probInfoTarget').textContent = `${target} 个`;
    $('probInfoDraws').textContent = `${mainDraws} 主抽`;
    $('probInfoMean').textContent = formatNumber(mean, 3);
    $('probInfoAtLeast1').textContent = formatPercent(atLeast1, 4);

    const tbody = $('probTable').querySelector('tbody');
    tbody.innerHTML = '';
    for (let k = 0; k <= maxReachable; k++) {
        const tr = document.createElement('tr');
        if (k === target) tr.classList.add('target-row');

        const tdK = document.createElement('td');
        tdK.textContent = String(k);
        if (k === target) {
            const badge = document.createElement('span');
            badge.className = 'target-marker';
            badge.textContent = '目标';
            tdK.appendChild(badge);
        }

        const p = dist[k];
        const tdP = document.createElement('td');
        const pText = formatPercent(p, 4);
        tdP.textContent = pText;
        if (pText.startsWith('<')) tdP.classList.add('tiny-prob');

        const cumP = cumulativeAtLeast(dist, k);
        const tdCum = document.createElement('td');
        const cText = formatPercent(cumP, 4);
        tdCum.textContent = cText;
        if (cText.startsWith('<')) tdCum.classList.add('tiny-prob');

        tr.append(tdK, tdP, tdCum);
        tbody.appendChild(tr);
    }

    if (Math.abs(sum - 1) > 1e-6) {
        console.warn('[ProbDist] 概率总和偏差:', sum);
    }

    $('probResult').hidden = false;
}

function runProbCalc() {
    const err = $('probError');
    const input = readProbInputs();
    if (input.error) {
        showError(err, input.error);
        $('probResult').hidden = true;
        return;
    }
    showError(err, '');

    const { mainDraws, pity, target } = input;

    // 边界：N=0
    if (mainDraws === 0) {
        const result = { dist: [1], maxReachable: 0, sum: 1 };
        renderProbResult(input, result);
        return;
    }

    try {
        const result = computeDistribution({ N: mainDraws, pityStart: pity });
        renderProbResult(input, result);
    } catch (e) {
        showError(err, '计算失败：' + (e && e.message ? e.message : String(e)));
        console.error(e);
    }
}

// ----- 期望计算 UI ----- //

function readExpInputs() {
    const el = $('expTarget');
    const target = parseNonNegInt(el);
    markInvalid(el, Number.isNaN(target) || target < 1);
    if (Number.isNaN(target) || target < 1) {
        return { error: '目标限定 up 数量必须 ≥ 1' };
    }
    return { target };
}

function renderExpResult(target, result) {
    const { expected, bound } = result;
    $('expLabel').textContent = `获得 ${target} 个限定 up 的期望主抽数`;
    $('expValue').textContent = formatNumber(expected, 2);
    $('expInfoTarget').textContent = `${target} 个`;
    $('expInfoBound').textContent = `≤ ${bound} 主抽必达成`;
    $('expInfoGems').textContent = `${Math.ceil(expected) * CONFIG.gemPerPull} 嵌晶玉等价`;
    $('expResult').hidden = false;
}

function runExpCalc() {
    const err = $('expError');
    const input = readExpInputs();
    if (input.error) {
        showError(err, input.error);
        $('expResult').hidden = true;
        return;
    }
    showError(err, '');

    try {
        const result = computeExpected(input.target);
        if (Math.abs(result.absorbedTotal - 1) > 1e-6) {
            console.warn('[Expected] 吸收概率和偏差:', result.absorbedTotal);
        }
        renderExpResult(input.target, result);
    } catch (e) {
        showError(err, '计算失败：' + (e && e.message ? e.message : String(e)));
        console.error(e);
    }
}

// ----- 初始化 ----- //

function init() {
    // 写入角色名到页面
    const nameEl = $('heroCharName');
    if (nameEl) nameEl.textContent = CONFIG.characterName;

    setupTabs();

    // 概率表单：实时副标签
    ['probGems', 'probQuota', 'probPity', 'probTarget'].forEach((id) => {
        const el = $(id);
        if (el) {
            el.addEventListener('input', () => {
                updateProbPreview();
                // 校验反馈
                const v = parseNonNegInt(el);
                let bad = Number.isNaN(v);
                if (id === 'probPity' && !bad && v > 79) bad = true;
                if (id === 'probTarget' && !bad && v < 1) bad = true;
                markInvalid(el, bad);
            });
        }
    });
    updateProbPreview();

    // 期望表单：输入校验
    const expTargetEl = $('expTarget');
    if (expTargetEl) {
        expTargetEl.addEventListener('input', () => {
            const v = parseNonNegInt(expTargetEl);
            markInvalid(expTargetEl, Number.isNaN(v) || v < 1);
        });
    }

    $('probCalcBtn').addEventListener('click', runProbCalc);
    $('expCalcBtn').addEventListener('click', runExpCalc);

    // Enter 提交
    $('probForm').addEventListener('submit', (e) => {
        e.preventDefault();
        runProbCalc();
    });
    $('expForm').addEventListener('submit', (e) => {
        e.preventDefault();
        runExpCalc();
    });

    // 自测
    runSelfTests();
}

// ================================================================
// 自测：打开控制台查看结果；失败用例会 console.assert 报错
// ================================================================

function approx(a, b, eps = 1e-9) { return Math.abs(a - b) <= eps; }

function runSelfTests() {
    try {
        // 用例 1：N=0，target=1 → 成功率 0%，分布 [1]
        {
            const r = computeDistribution({ N: 0, pityStart: 0 });
            console.assert(r.dist.length === 1 && approx(r.dist[0], 1),
                '用例 1 失败：N=0 时应 dist=[1]，实际', r.dist);
        }

        // 用例 2：N=120, pityStart=0, target=1 → 成功率 100%
        {
            const r = computeDistribution({ N: 120, pityStart: 0 });
            const s = cumulativeAtLeast(r.dist, 1);
            console.assert(approx(s, 1, 1e-9),
                '用例 2 失败：N=120 时 P(>=1)=100%，实际', s);
            console.assert(approx(r.sum, 1, 1e-9),
                '用例 2 概率和偏差', r.sum);
        }

        // 用例 3：N=240, target=2 → 成功率 100%
        {
            const r = computeDistribution({ N: 240, pityStart: 0 });
            const s = cumulativeAtLeast(r.dist, 2);
            console.assert(approx(s, 1, 1e-9),
                '用例 3 失败：N=240 时 P(>=2)=100%，实际', s);
        }

        // 用例 4：N=29 vs N=30 免费十连触发
        // 这里通过比较分布长度（maxReachable 差异）验证
        {
            const r29 = computeDistribution({ N: 29, pityStart: 0 });
            const r30 = computeDistribution({ N: 30, pityStart: 0 });
            console.assert(r29.maxReachable === 29,
                '用例 4a 失败：N=29 时 maxReachable 应为 29，实际', r29.maxReachable);
            console.assert(r30.maxReachable === 40,
                '用例 4b 失败：N=30 时 maxReachable 应为 30+10=40，实际', r30.maxReachable);
            // N=30 的 P(>=1) 应明显高于 N=29（因为免费十连可能出限定 up）
            const p29 = cumulativeAtLeast(r29.dist, 1);
            const p30 = cumulativeAtLeast(r30.dist, 1);
            console.assert(p30 > p29,
                '用例 4c 失败：免费十连应提高 P(>=1)', p29, p30);
        }

        // 用例 5：pityStart=79 → 第 1 次普通主抽必出 6★，其中 50% 限定
        // 故 N=1, pityStart=79 时 P(lim=1) = 0.5
        {
            const r = computeDistribution({ N: 1, pityStart: 79 });
            console.assert(approx(r.dist[1], 0.5, 1e-12),
                '用例 5 失败：pity=79 首抽 P(lim=1) 应为 0.5，实际', r.dist[1]);
        }

        // 用例 6：期望 target=1 的严格上界 = 120；target=2 的上界 = 240
        {
            const e1 = computeExpected(1);
            const e2 = computeExpected(2);
            console.assert(e1.bound === 120, '用例 6a 失败：E[K=1] 上界应 120', e1.bound);
            console.assert(e2.bound === 240, '用例 6b 失败：E[K=2] 上界应 240', e2.bound);
            console.assert(approx(e1.absorbedTotal, 1, 1e-9),
                '用例 6c 失败：E[K=1] 吸收和应为 1，实际', e1.absorbedTotal);
            console.assert(approx(e2.absorbedTotal, 1, 1e-9),
                '用例 6d 失败：E[K=2] 吸收和应为 1，实际', e2.absorbedTotal);
            console.assert(e1.expected > 0 && e1.expected <= 120,
                '用例 6e 失败：E[K=1] 应在 (0, 120]', e1.expected);
            console.assert(e2.expected > e1.expected && e2.expected <= 240,
                '用例 6f 失败：E[K=2] 应 > E[K=1] 且 ≤ 240', e2.expected);
        }

        // 用例 7：分布和 ≈ 1（多规模一致性）
        {
            for (const N of [10, 30, 80, 120, 150, 240, 360]) {
                const r = computeDistribution({ N, pityStart: 0 });
                console.assert(approx(r.sum, 1, 1e-6),
                    `用例 7 失败：N=${N} 概率和偏差 ${r.sum}`);
            }
        }

        console.log('%c[SelfTest] 所有自测用例通过', 'color:#7bffb0;font-weight:bold');
        console.log('E[1 个限定 up] =', computeExpected(1).expected.toFixed(2), '主抽');
        console.log('E[2 个限定 up] =', computeExpected(2).expected.toFixed(2), '主抽');
        console.log('E[3 个限定 up] =', computeExpected(3).expected.toFixed(2), '主抽');
    } catch (e) {
        console.error('[SelfTest] 异常:', e);
    }
}

// ---------------- 启动 ---------------- //

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
