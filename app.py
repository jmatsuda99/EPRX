
# zero-aligned dual y-axis example (to be merged into app.py)

def zero_ratio(ymin, ymax):
    if ymin >= 0:
        return 0.0
    if ymax <= 0:
        return 1.0
    return abs(ymin) / (ymax - ymin)

def aligned_ylim(ymin, ymax, r):
    span = max(abs(ymin), abs(ymax))
    lower = -r * span / max(r, 1 - r)
    upper = (1 - r) * span / max(r, 1 - r)
    return lower, upper

# after plotting on ax1 / ax2
cf_min, cf_max = min(yearly_cf), max(yearly_cf)
cum_min, cum_max = min(cum_cf), max(cum_cf)

r = max(zero_ratio(cf_min, cf_max), zero_ratio(cum_min, cum_max))

ax1.set_ylim(*aligned_ylim(cf_min, cf_max, r))
ax2.set_ylim(*aligned_ylim(cum_min, cum_max, r))

ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
