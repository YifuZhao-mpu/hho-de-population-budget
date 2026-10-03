# SEHHO-COBL revision: statistical analysis
All p-values are Holm step-down corrected within their family.
Effect size is Cliff's delta of the control against the competitor (negative favours SEHHO-COBL); A12 is the probability the control produces the smaller error.


## CEC2022, paper budget (15,000 FEs), strictly FE-matched

### CEC2022, paper budget (15,000 FEs), strictly FE-matched - D=10
Friedman over 12 functions x 6 algorithms: chi2=37.43, Iman-Davenport F=18.24, p=1.242e-10

 algorithm  avg_rank  rank1_count  position
    LSHADE  2.083333            3         1
       jSO  2.250000            4         2
SEHHO-COBL  2.500000            4         3
     SHADE  4.083333            1         4
        DE  4.250000            0         5
       HHO  5.833333            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z        p   p_holm  significant_holm win_tie_loss  mean_cliffs_delta magnitude
       DE  4.250000 -2.291288 0.021947 0.087787             False        7/5/0          -0.687778     large
      HHO  5.833333 -4.364358 0.000013 0.000064              True       12/0/0          -0.933889     large
   LSHADE  2.083333  0.545545 0.585379 1.000000             False        3/7/2          -0.164074     small
    SHADE  4.083333 -2.073070 0.038166 0.114497             False        9/3/0          -0.729630     large
      jSO  2.250000  0.327327 0.743421 1.000000             False        5/5/2          -0.264074     small

### CEC2022, paper budget (15,000 FEs), strictly FE-matched - D=20
Friedman over 12 functions x 6 algorithms: chi2=47.90, Iman-Davenport F=43.57, p=6.253e-18

 algorithm  avg_rank  rank1_count  position
SEHHO-COBL  1.833333            7         1
       jSO  2.083333            3         2
    LSHADE  2.166667            2         3
     SHADE  4.250000            0         4
        DE  5.083333            0         5
       HHO  5.583333            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z            p   p_holm  significant_holm win_tie_loss  mean_cliffs_delta magnitude
       DE  5.083333 -4.255249 2.088166e-05 0.000084              True       12/0/0          -0.947037     large
      HHO  5.583333 -4.909903 9.112167e-07 0.000005              True       12/0/0          -0.992778     large
   LSHADE  2.166667 -0.436436 6.625206e-01 1.000000             False        9/1/2          -0.536481     large
    SHADE  4.250000 -3.164159 1.555316e-03 0.004666              True       11/1/0          -0.832963     large
      jSO  2.083333 -0.327327 7.434207e-01 1.000000             False        8/2/2          -0.415926    medium


## CEC2022, competition budget

### CEC2022, competition budget - D=10
Friedman over 12 functions x 6 algorithms: chi2=33.43, Iman-Davenport F=13.84, p=9.408e-09

 algorithm  avg_rank  rank1_count  position
    LSHADE  2.416667            2         1
       jSO  2.416667            4         2
     SHADE  2.916667            1         3
        DE  3.000000            2         4
SEHHO-COBL  4.250000            3         5
       HHO  6.000000            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z        p   p_holm  significant_holm win_tie_loss  mean_cliffs_delta  magnitude
       DE  3.000000  1.636634 0.101707 0.161711             False        1/9/2           0.140833 negligible
      HHO  6.000000 -2.291288 0.021947 0.081887             False       11/1/0          -0.929259      large
   LSHADE  2.416667  2.400397 0.016377 0.081887             False        0/8/4           0.280185      small
    SHADE  2.916667  1.745743 0.080856 0.161711             False        2/7/3           0.128333 negligible
      jSO  2.416667  2.400397 0.016377 0.081887             False        0/8/4           0.251759      small

### CEC2022, competition budget - D=20
Friedman over 12 functions x 6 algorithms: chi2=33.33, Iman-Davenport F=13.75, p=1.034e-08

 algorithm  avg_rank  rank1_count  position
       jSO  2.041667            2         1
    LSHADE  2.291667            3         2
        DE  3.291667            4         3
     SHADE  3.458333            0         4
SEHHO-COBL  4.000000            3         5
       HHO  5.916667            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z        p   p_holm  significant_holm win_tie_loss  mean_cliffs_delta magnitude
       DE  3.291667  0.927426 0.353705 0.707411             False        0/7/5           0.350185    medium
      HHO  5.916667 -2.509506 0.012090 0.051728             False       12/0/0          -0.964444     large
   LSHADE  2.291667  2.236733 0.025304 0.075911             False        0/4/8           0.607778     large
    SHADE  3.458333  0.709208 0.478195 0.707411             False        0/6/6           0.462778    medium
      jSO  2.041667  2.564060 0.010346 0.051728             False        0/4/8           0.638704     large


## CEC2017, competition budget (10,000 x D)

### CEC2017, competition budget (10,000 x D) - D=30
Friedman over 29 functions x 6 algorithms: chi2=108.70, Iman-Davenport F=83.86, p=2.317e-40

 algorithm  avg_rank  rank1_count  position
       jSO  1.534483           16         1
    LSHADE  2.120690            8         2
     SHADE  2.948276            1         3
        DE  4.051724            2         4
SEHHO-COBL  4.413793            2         5
       HHO  5.931034            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z            p       p_holm  significant_holm win_tie_loss  mean_cliffs_delta  magnitude
       DE  4.051724  0.736956 4.611494e-01 4.611494e-01             False      11/6/12           0.028429 negligible
      HHO  5.931034 -3.088195 2.013766e-03 6.041297e-03              True       28/1/0          -0.935785      large
   LSHADE  2.120690  4.667385 3.050574e-06 1.220230e-05              True       0/5/24           0.804215      large
    SHADE  2.948276  2.982915 2.855171e-03 6.041297e-03              True       0/3/26           0.750805      large
      jSO  1.534483  5.860551 4.613336e-09 2.306668e-08              True       0/4/25           0.847931      large

### CEC2017, competition budget (10,000 x D) - D=50
Friedman over 29 functions x 6 algorithms: chi2=112.01, Iman-Davenport F=95.07, p=3.028e-43

 algorithm  avg_rank  rank1_count  position
       jSO  1.500000           15         1
    LSHADE  1.913793           11         2
     SHADE  3.172414            0         3
        DE  4.241379            2         4
SEHHO-COBL  4.275862            1         5
       HHO  5.896552            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z            p       p_holm  significant_holm win_tie_loss  mean_cliffs_delta magnitude
       DE  4.241379  0.070186 9.440454e-01 9.440454e-01             False      16/3/10          -0.190728     small
      HHO  5.896552 -3.298753 9.711522e-04 2.913456e-03              True       29/0/0          -0.976245     large
   LSHADE  1.913793  4.807757 1.526328e-06 6.105311e-06              True       0/4/25           0.818199     large
    SHADE  3.172414  2.245960 2.470659e-02 4.941318e-02              True       1/5/23           0.696973     large
      jSO  1.500000  5.649992 1.604550e-08 8.022748e-08              True       1/2/26           0.852720     large

### CEC2017, competition budget (10,000 x D) - D=100
Friedman over 29 functions x 6 algorithms: chi2=114.22, Iman-Davenport F=103.91, p=2.421e-45

 algorithm  avg_rank  rank1_count  position
       jSO  1.448276           17         1
    LSHADE  2.000000            9         2
     SHADE  3.000000            3         3
SEHHO-COBL  4.068966            0         4
        DE  4.655172            0         5
       HHO  5.827586            0         6

Post-hoc vs SEHHO-COBL (Holm-corrected) and per-function rank-sum win/tie/loss (Holm-corrected across the whole family):
algorithm  avg_rank         z            p       p_holm  significant_holm win_tie_loss  mean_cliffs_delta magnitude
       DE  4.655172 -1.193166 2.328043e-01 2.328043e-01             False      16/11/2          -0.418544    medium
      HHO  5.827586 -3.579498 3.442545e-04 1.032764e-03              True       28/1/0          -0.972107     large
   LSHADE  2.000000  4.211174 2.540465e-05 1.016186e-04              True       2/2/25           0.783448     large
    SHADE  3.000000  2.175773 2.957220e-02 5.914441e-02             False       1/2/26           0.750115     large
      jSO  1.448276  5.334154 9.599088e-08 4.799544e-07              True       1/3/25           0.809808     large


## Ablations on CEC2022 (15,000 FEs)

### D=10  (Friedman chi2=21.02, Iman-Davenport F=2.08, p=2.642e-02)
                 variant  avg_rank  wins  losses        p   p_holm  significant_holm  cliffs_delta  magnitude      a12
SEHHO:AlwaysPbest+NoLevy  4.750000     4       6 0.130859 1.000000             False      0.101759 negligible 0.449120
            SEHHO:NoLevy  4.833333     3       7 0.556641 1.000000             False      0.081111 negligible 0.459444
SEHHO:AlwaysPbest+FixedP  5.250000     4       6 0.160156 1.000000             False     -0.035185 negligible 0.517593
       SEHHO:AlwaysPbest  5.666667     3       7 0.193359 1.000000             False     -0.018519 negligible 0.509259
SEHHO:DeterministicPhase  6.000000     5       6 0.365234 1.000000             False      0.011481 negligible 0.494259
            SEHHO:FixedP  6.458333     6       4 0.492188 1.000000             False      0.057870 negligible 0.471065
    SEHHO:NoEscapeEnergy  6.750000     7       3 0.375000 1.000000             False      0.010926 negligible 0.494537
         SEHHO:NoArchive  6.916667     6       4 0.695312 1.000000             False     -0.004907 negligible 0.502454
            SEHHO:NoCOBL  7.208333     6       4 0.556641 1.000000             False      0.016296 negligible 0.491852
        SEHHO:AlwaysBest  8.083333     6       4 0.193359 1.000000             False     -0.005741 negligible 0.502870
           SEHHO:NoSHADE  9.833333     9       2 0.032227 0.354492             False     -0.373889     medium 0.686944
(full-method average rank = 6.25)

### D=20  (Friedman chi2=18.44, Iman-Davenport F=1.79, p=6.356e-02)
                 variant  avg_rank  wins  losses        p   p_holm  significant_holm  cliffs_delta  magnitude      a12
SEHHO:AlwaysPbest+NoLevy  4.250000     4       8 0.791016 1.000000             False     -0.072130 negligible 0.536065
SEHHO:AlwaysPbest+FixedP  5.000000     6       6 0.677246 1.000000             False     -0.185556      small 0.592778
            SEHHO:FixedP  5.750000     7       5 0.423828 1.000000             False     -0.073056 negligible 0.536528
SEHHO:DeterministicPhase  5.833333     4       8 0.569336 1.000000             False     -0.157870      small 0.578935
            SEHHO:NoCOBL  6.333333     6       6 0.850098 1.000000             False     -0.039722 negligible 0.519861
            SEHHO:NoLevy  6.333333     5       7 1.000000 1.000000             False     -0.005926 negligible 0.502963
       SEHHO:AlwaysPbest  6.500000     6       6 0.469727 1.000000             False     -0.214259      small 0.607130
         SEHHO:NoArchive  7.333333     6       6 0.622070 1.000000             False     -0.252685      small 0.626343
        SEHHO:AlwaysBest  7.416667     7       5 0.622070 1.000000             False     -0.043241 negligible 0.521620
    SEHHO:NoEscapeEnergy  7.500000     8       4 0.077148 0.771484             False     -0.051574 negligible 0.525787
           SEHHO:NoSHADE  9.500000    10       2 0.063965 0.703613             False     -0.654074      large 0.827037
(full-method average rank = 6.25)


## Ablations on CEC2017 D=30 (300,000 FEs)

### D=30  (Friedman chi2=123.81, Iman-Davenport F=17.76, p=3.005e-27)
                 variant  avg_rank  wins  losses        p   p_holm  significant_holm  cliffs_delta  magnitude      a12
       SEHHO:AlwaysPbest  3.689655     4      24 0.000526 0.005256              True      0.166092      small 0.416954
SEHHO:DeterministicPhase  4.034483     5      23 0.007629 0.061033             False      0.152261      small 0.423870
SEHHO:AlwaysPbest+FixedP  4.137931     5      23 0.016728 0.117094             False      0.129693 negligible 0.435153
SEHHO:AlwaysPbest+NoLevy  4.448276     8      20 0.023224 0.139345             False      0.151916      small 0.424042
         SEHHO:NoArchive  6.000000    12      16 0.316096 1.000000             False     -0.002146 negligible 0.501073
            SEHHO:FixedP  6.379310    13      15 0.551874 1.000000             False      0.023372 negligible 0.488314
            SEHHO:NoLevy  6.758621    13      15 0.479096 1.000000             False     -0.000115 negligible 0.500057
            SEHHO:NoCOBL  7.551724    14      14 0.796562 1.000000             False     -0.008812 negligible 0.504406
    SEHHO:NoEscapeEnergy  7.862069    19       9 0.119971 0.599857             False     -0.122146 negligible 0.561073
        SEHHO:AlwaysBest  9.034483    22       6 0.003739 0.033648              True     -0.186360      small 0.593180
           SEHHO:NoSHADE 11.172414    27       2 0.000003 0.000037              True     -0.683525      large 0.841762
(full-method average rank = 6.93)


## Constrained engineering problems (feasibility rule)

                 problem  algorithm  runs  feasible_runs  feasibility_rate  best_feasible  mean_feasible  std_feasible  mean_total_violation  max_total_violation  mean_max_violation
      P1_pressure_vessel         DE    30             30             100.0   6.059714e+03   6.059716e+03  1.768413e-03          6.882708e-10         9.133822e-09        6.882708e-10
      P1_pressure_vessel        HHO    30             30             100.0   6.107066e+03   6.914024e+03  4.810484e+02          2.398156e-09         9.778887e-09        2.398156e-09
      P1_pressure_vessel     LSHADE    30             30             100.0   6.059714e+03   6.087641e+03  7.807406e+01          1.000000e-08         1.000000e-08        1.000000e-08
      P1_pressure_vessel SEHHO-COBL    30             30             100.0   6.059714e+03   6.260699e+03  2.520703e+02          1.000000e-08         1.000000e-08        1.000000e-08
      P1_pressure_vessel      SHADE    30             30             100.0   6.060676e+03   6.096775e+03  7.631718e+01          0.000000e+00         0.000000e+00        0.000000e+00
      P1_pressure_vessel        jSO    30             30             100.0   6.059714e+03   6.059714e+03  1.850085e-12          1.000000e-08         1.000000e-08        1.000000e-08
           P2_gear_train         DE    30             30             100.0   2.700857e-12   5.831239e-11  1.621969e-10          0.000000e+00         0.000000e+00        0.000000e+00
           P2_gear_train        HHO    30             30             100.0   2.700857e-12   4.173943e-09  6.182182e-09          0.000000e+00         0.000000e+00        0.000000e+00
           P2_gear_train     LSHADE    30             30             100.0   2.700857e-12   1.221026e-11  1.033977e-11          0.000000e+00         0.000000e+00        0.000000e+00
           P2_gear_train SEHHO-COBL    30             30             100.0   2.700857e-12   6.244044e-11  1.817975e-10          0.000000e+00         0.000000e+00        0.000000e+00
           P2_gear_train      SHADE    30             30             100.0   2.700857e-12   1.867077e-11  5.843343e-11          0.000000e+00         0.000000e+00        0.000000e+00
           P2_gear_train        jSO    30             30             100.0   2.700857e-12   6.097074e-12  7.723996e-12          0.000000e+00         0.000000e+00        0.000000e+00
  P3_heat_exchanger_RC01         DE    30              0               0.0            NaN            NaN           NaN          2.094195e+02         1.173272e+03        9.599886e+01
  P3_heat_exchanger_RC01        HHO    30              0               0.0            NaN            NaN           NaN          9.885375e+05         2.000028e+06        7.750746e+05
  P3_heat_exchanger_RC01     LSHADE    30              0               0.0            NaN            NaN           NaN          8.469815e+01         9.966298e+02        5.324009e+01
  P3_heat_exchanger_RC01 SEHHO-COBL    30             12              40.0   2.111243e+02   5.395303e+02  1.623807e+02          6.367105e+00         1.318323e+02        5.530664e+00
  P3_heat_exchanger_RC01      SHADE    30              0               0.0            NaN            NaN           NaN          3.346481e+03         7.143700e+03        1.379475e+03
  P3_heat_exchanger_RC01        jSO    30              0               0.0            NaN            NaN           NaN          1.961620e+00         2.868030e+01        1.591805e+00
P4_blending_pooling_RC06         DE    30              0               0.0            NaN            NaN           NaN          5.152348e+01         8.035373e+01        1.057795e+01
P4_blending_pooling_RC06        HHO    30              0               0.0            NaN            NaN           NaN          1.101047e+02         2.345337e+02        6.743815e+01
P4_blending_pooling_RC06     LSHADE    30              0               0.0            NaN            NaN           NaN          2.747671e+01         4.044657e+01        9.306749e+00
P4_blending_pooling_RC06 SEHHO-COBL    30              0               0.0            NaN            NaN           NaN          1.376135e+01         2.759298e+01        7.776551e+00
P4_blending_pooling_RC06      SHADE    30              0               0.0            NaN            NaN           NaN          3.381606e+01         4.017451e+01        1.073706e+01
P4_blending_pooling_RC06        jSO    30              0               0.0            NaN            NaN           NaN          1.698475e+01         2.494874e+01        7.529175e+00
     P5_batch_plant_RC14         DE    30             30             100.0   5.372839e+04   5.446527e+04  1.886185e+03          0.000000e+00         0.000000e+00        0.000000e+00
     P5_batch_plant_RC14        HHO    30             30             100.0   6.618442e+04   7.840500e+04  9.987651e+03          6.660224e-09         9.999999e-09        6.660224e-09
     P5_batch_plant_RC14     LSHADE    30             30             100.0   5.850566e+04   5.851078e+04  5.090624e+00          0.000000e+00         0.000000e+00        0.000000e+00
     P5_batch_plant_RC14 SEHHO-COBL    30             30             100.0   5.363894e+04   5.806265e+04  1.505845e+03          2.549391e-09         9.971403e-09        2.455574e-09
     P5_batch_plant_RC14      SHADE    30             30             100.0   5.874259e+04   5.892753e+04  1.007376e+02          0.000000e+00         0.000000e+00        0.000000e+00
     P5_batch_plant_RC14        jSO    30             30             100.0   5.850552e+04   5.850682e+04  1.138526e+00          1.072975e-10         3.218924e-09        1.072975e-10

**P4_blending_pooling_RC06: no algorithm produced a single feasible design in any run at this budget.**