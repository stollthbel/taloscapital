(* Trident 1M Scalper — Tick-Based ODE Model with Reinforced Learning and Fibonacci Logic *)

(* — Types — *)
type tick = {
  bid: float;
  ask: float;
  last: float;
  volume: float;
  timestamp: float;
}

type state = {
  ema_fast: float;
  ema_slow: float;
  vwap: float;
  price: float;
  momentum: float;
  rsi: float;
  atr: float;
  dprice_dt: float;
  dp_dt: float;
  gain: float;
  loss: float;
  tr: float;
  rs: float;
  vol_ema: float;
  price_volatility: float;
  take_profit: float;
  stop_loss: float;
  fitness: float;
  fib_support: float;
  fib_resistance: float;
  acceleration: float;
  jerk: float;
}

(* — Dynamic Parameters — *)
type genome = {
  alpha_fast: float;
  alpha_slow: float;
  rsi_threshold: float;
  take_profit_mult: float;
  stop_loss_mult: float;
  fib_ratio1: float;
  fib_ratio2: float;
  reward_risk_weight: float;
  ode_accel_coeff: float;
  ode_jerk_coeff: float;
}

let default_genome = {
  alpha_fast = 2. /. 5.;
  alpha_slow = 2. /. 21.;
  rsi_threshold = 25.;
  take_profit_mult = 1.5;
  stop_loss_mult = 0.75;
  fib_ratio1 = 0.236;
  fib_ratio2 = 0.764;
  reward_risk_weight = 2.0;
  ode_accel_coeff = 0.15;
  ode_jerk_coeff = 0.02;
}

let ema_ode alpha ema price = alpha *. (price -. ema)

let rsi_ode gain loss =
  let rs = if loss = 0.0 then 1000.0 else gain /. loss in
  100.0 -. (100.0 /. (1.0 +. rs)), rs

let vwap_ode cum_pv cum_vol price volume =
  let tpv = price *. volume in
  let new_cum_pv = cum_pv +. tpv in
  let new_cum_vol = cum_vol +. volume in
  new_cum_pv /. new_cum_vol, new_cum_pv, new_cum_vol

let true_range prev_close price high low =
  let tr1 = high -. low in
  let tr2 = abs_float (high -. prev_close) in
  let tr3 = abs_float (low -. prev_close) in
  max tr1 (max tr2 tr3)

let atr_ode prev_atr tr alpha = alpha *. tr +. (1.0 -. alpha) *. prev_atr

let volatility_ode prev_vol delta alpha = alpha *. (delta *. delta) +. (1. -. alpha) *. prev_vol

let fib_levels high low ratio1 ratio2 =
  let diff = high -. low in
  low +. ratio1 *. diff, high -. ratio2 *. diff

let fitness_function reward risk reward_risk_weight =
  if risk = 0.0 then reward *. reward_risk_weight
  else reward /. risk *. reward_risk_weight

let state_ode genome t (st: state) (tick: tick) (prev_close: float) (cum_pv, cum_vol) high low =
  let d_ema_fast = ema_ode genome.alpha_fast st.ema_fast tick.last in
  let d_ema_slow = ema_ode genome.alpha_slow st.ema_slow tick.last in
  let new_vwap, new_pv, new_vol = vwap_ode cum_pv cum_vol tick.last tick.volume in
  let price = tick.last in
  let delta = price -. prev_close in
  let gain = if delta > 0.0 then delta else 0.0 in
  let loss = if delta < 0.0 then -.delta else 0.0 in
  let new_gain = 0.2 *. gain +. 0.8 *. st.gain in
  let new_loss = 0.2 *. loss +. 0.8 *. st.loss in
  let new_rsi, rs = rsi_ode new_gain new_loss in
  let tr = true_range prev_close price high low in
  let new_tr = 0.2 *. tr +. 0.8 *. st.tr in
  let new_atr = atr_ode st.atr new_tr (2. /. 9.) in
  let new_vol_ema = ema_ode (2. /. 13.) st.vol_ema tick.volume in
  let new_volatility = volatility_ode st.price_volatility delta (2. /. 13.) in
  let acceleration = genome.ode_accel_coeff *. (delta -. st.dprice_dt) +. (1. -. genome.ode_accel_coeff) *. st.acceleration in
  let jerk = genome.ode_jerk_coeff *. (acceleration -. st.acceleration) +. (1. -. genome.ode_jerk_coeff) *. st.jerk in
  let dp_dt = st.dprice_dt +. acceleration +. 0.1 *. (st.ema_fast -. price) +. 0.05 *. (st.vwap -. price) +. 0.01 *. (new_volatility -. st.price_volatility) in
  let take_profit = price +. genome.take_profit_mult *. new_atr in
  let stop_loss = price -. genome.stop_loss_mult *. new_atr in
  let fib_supp, fib_res = fib_levels high low genome.fib_ratio1 genome.fib_ratio2 in
  let reward = take_profit -. price in
  let risk = price -. stop_loss in
  let fitness = fitness_function reward risk genome.reward_risk_weight in
  {
    ema_fast = st.ema_fast +. d_ema_fast;
    ema_slow = st.ema_slow +. d_ema_slow;
    vwap = new_vwap;
    price = price +. st.dprice_dt;
    momentum = dp_dt;
    rsi = new_rsi;
    atr = new_atr;
    dprice_dt = dp_dt;
    dp_dt = 0.0;
    gain = new_gain;
    loss = new_loss;
    tr = new_tr;
    rs = rs;
    vol_ema = new_vol_ema;
    price_volatility = new_volatility;
    take_profit = take_profit;
    stop_loss = stop_loss;
    fitness = fitness;
    fib_support = fib_supp;
    fib_resistance = fib_res;
    acceleration = acceleration;
    jerk = jerk;
  }, (new_pv, new_vol)

(* Signal generation remains unchanged *)

let () =
  let fake_ticks = [] in
  let signals = generate_signals fake_ticks default_genome in
  List.iter (fun (dir, t, px, tp, sl, f) ->
    Printf.printf "%s at t=%.1f, price=%.2f, TP=%.2f, SL=%.2f, Fitness=%.3f\n" dir t px tp sl f
  ) signals
