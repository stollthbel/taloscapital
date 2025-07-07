(* Trident Alpha Model — Higher-Order ODE Price/Signal Interaction Model with Genetic Reinforcement Learning and Fibonacci Adaptive Reward Logic *)

(* — Types — *)
type bar = {
  open_price: float;
  high_price: float;
  low_price: float;
  close_price: float;
  volume: float
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
  alpha_fast = 2. /. 10.;
  alpha_slow = 2. /. 50.;
  rsi_threshold = 30.;
  take_profit_mult = 2.0;
  stop_loss_mult = 1.0;
  fib_ratio1 = 0.382;
  fib_ratio2 = 0.618;
  reward_risk_weight = 1.5;
  ode_accel_coeff = 0.1;
  ode_jerk_coeff = 0.01;
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

let true_range prev_close bar =
  let tr1 = bar.high_price -. bar.low_price in
  let tr2 = abs_float (bar.high_price -. prev_close) in
  let tr3 = abs_float (bar.low_price -. prev_close) in
  max tr1 (max tr2 tr3)

let atr_ode prev_atr tr alpha = alpha *. tr +. (1.0 -. alpha) *. prev_atr

let volatility_ode prev_vol delta alpha = alpha *. (delta *. delta) +. (1. -. alpha) *. prev_vol

let fib_levels high low ratio1 ratio2 =
  let diff = high -. low in
  low +. ratio1 *. diff, high -. ratio2 *. diff

let fitness_function reward risk reward_risk_weight =
  if risk = 0.0 then reward *. reward_risk_weight
  else reward /. risk *. reward_risk_weight

let state_ode genome t (st: state) (bar: bar) (prev_close: float) (cum_pv, cum_vol) =
  let d_ema_fast = ema_ode genome.alpha_fast st.ema_fast bar.close_price in
  let d_ema_slow = ema_ode genome.alpha_slow st.ema_slow bar.close_price in
  let new_vwap, new_pv, new_vol = vwap_ode cum_pv cum_vol bar.close_price bar.volume in
  let price = bar.close_price in
  let delta = price -. prev_close in
  let gain = if delta > 0.0 then delta else 0.0 in
  let loss = if delta < 0.0 then -.delta else 0.0 in
  let new_gain = 0.1 *. gain +. 0.9 *. st.gain in
  let new_loss = 0.1 *. loss +. 0.9 *. st.loss in
  let new_rsi, rs = rsi_ode new_gain new_loss in
  let tr = true_range prev_close bar in
  let new_tr = 0.1 *. tr +. 0.9 *. st.tr in
  let new_atr = atr_ode st.atr new_tr (2. /. 15.) in
  let new_vol_ema = ema_ode (2. /. 21.) st.vol_ema bar.volume in
  let new_volatility = volatility_ode st.price_volatility delta (2. /. 21.) in
  let acceleration = genome.ode_accel_coeff *. (delta -. st.dprice_dt) +. (1. -. genome.ode_accel_coeff) *. st.acceleration in
  let jerk = genome.ode_jerk_coeff *. (acceleration -. st.acceleration) +. (1. -. genome.ode_jerk_coeff) *. st.jerk in
  let dp_dt = st.dprice_dt +. acceleration +. 0.1 *. (st.ema_fast -. price) +. 0.05 *. (st.vwap -. price) +. 0.01 *. (new_volatility -. st.price_volatility) in
  let take_profit = price +. genome.take_profit_mult *. new_atr in
  let stop_loss = price -. genome.stop_loss_mult *. new_atr in
  let fib_supp, fib_res = fib_levels bar.high_price bar.low_price genome.fib_ratio1 genome.fib_ratio2 in
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

let generate_signals bars genome =
  let rec loop acc st t cum_pv cum_vol = function
    | [] -> List.rev acc
    | bar::rest ->
      let prev_close = st.price in
      let st', (new_pv, new_vol) = state_ode genome t st bar prev_close (cum_pv, cum_vol) in
      let long_signal =
        st'.ema_fast > st'.ema_slow &&
        st'.price > st'.vwap &&
        st'.rsi < genome.rsi_threshold &&
        st'.momentum > 0.0 &&
        st'.vol_ema > st.vol_ema &&
        st'.price > st'.fib_support
      in
      let short_signal =
        st'.ema_fast < st'.ema_slow &&
        st'.price < st'.vwap &&
        st'.rsi > (100. -. genome.rsi_threshold) &&
        st'.momentum < 0.0 &&
        st'.vol_ema < st.vol_ema &&
        st'.price < st'.fib_resistance
      in
      let signal =
        if long_signal then Some ("LONG", t, st'.price, st'.take_profit, st'.stop_loss, st'.fitness)
        else if short_signal then Some ("SHORT", t, st'.price, st'.take_profit, st'.stop_loss, st'.fitness)
        else None
      in
      loop (match signal with Some s -> s::acc | None -> acc) st' (t+.1.) new_pv new_vol rest
  in
  match bars with
  | [] -> []
  | b0::bs ->
    let init_price = b0.close_price in
    let init_state = {
      ema_fast = init_price;
      ema_slow = init_price;
      vwap = init_price;
      price = init_price;
      momentum = 0.0;
      rsi = 50.0;
      atr = 0.0;
      dprice_dt = 0.0;
      dp_dt = 0.0;
      gain = 0.0;
      loss = 0.0;
      tr = 0.0;
      rs = 1.0;
      vol_ema = 0.0;
      price_volatility = 0.0;
      take_profit = init_price *. 1.02;
      stop_loss = init_price *. 0.98;
      fitness = 0.0;
      fib_support = init_price *. 0.9;
      fib_resistance = init_price *. 1.1;
      acceleration = 0.0;
      jerk = 0.0;
    } in
    loop [] init_state 0.0 0.0 0.0 (b0::bs)

let () =
  let fake_bars = [] in
  let signals = generate_signals fake_bars default_genome in
  List.iter (fun (dir, t, px, tp, sl, f) ->
    Printf.printf "%s at t=%.1f, price=%.2f, TP=%.2f, SL=%.2f, Fitness=%.3f\n" dir t px tp sl f
  ) signals
