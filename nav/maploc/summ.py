import sys, json
for f in sys.argv[1:]:
    for n, r in json.load(open(f)).items():
        print(f.split('/')[0], n, 'odom %.3f/%.3f/%.3f' % (r['odom']['pos_med'], r['odom']['pos_p90'], r['odom']['pos_final']), 's2 %.3f/%.3f/%.3f' % (r['slam2']['pos_med'], r['slam2']['pos_p90'], r['slam2']['pos_final']),
              'ml %.3f/%.3f/%.3f y90 %.2f' % (r['maploc']['pos_med'], r['maploc']['pos_p90'], r['maploc']['pos_final'], r['maploc']['yaw_p90']), 's %.4f+-%.4f' % (r['map_scale_final'], r['map_scale_sd']),
              'pitch %s' % (None if r.get('pitch_deg') is None else '%.2f+-%.2f' % (r['pitch_deg'], r['pitch_sd_deg'])), 'rej %.2f' % r['reject_rate'], 'ms %.0f' % r['ms_med'])
