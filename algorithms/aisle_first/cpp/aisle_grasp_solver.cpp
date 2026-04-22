// Aisle-centric GRASP solver for the SBPO 2025 wave order-picking problem.
//
// CLI:
//   aisle_grasp_solver <instance_path>
//       --alpha=F
//       --scoring=static|adaptive
//       --aisle-score=useful|units|variety|mixed
//       --packing-order=asc|desc|shuffle
//       --max-iterations=N
//       --greedy=simple|multi
//       --local-search-aisle=none|swap|full
//       --local-search-order=none|swap|full
//       [--seed=N]
//       [--time-limit=F]         seconds; 0 or omitted = unlimited
//
// Output (stdout): one JSON object per line. Each time the best-known
// solution improves, a line is emitted immediately (and flushed) with:
//   {"objective":F,"selected_orders":[...],"visited_aisles":[...]}
// The first line is an initial empty incumbent; the final line repeats
// the terminal incumbent so the consumer always has a valid record.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Scoring { Static, Adaptive };
enum class AisleScore { Useful, Units, Variety, Mixed };
enum class PackOrder { Asc, Desc, Shuffle };
enum class Greedy { Simple, Multi };
enum class LocalSearch { None, Swap, Full };

struct Options {
    std::string instance_path;
    double alpha = 0.0;
    Scoring scoring = Scoring::Static;
    AisleScore aisle_score = AisleScore::Useful;
    PackOrder packing_order = PackOrder::Desc;
    int max_iterations = 0;
    Greedy greedy = Greedy::Simple;
    LocalSearch local_search_aisle = LocalSearch::None;
    LocalSearch local_search_order = LocalSearch::None;
    bool seed_set = false;
    uint64_t seed = 0;
    double time_limit = 0.0;
};

struct Problem {
    int n_orders = 0;
    int n_items = 0;
    int n_aisles = 0;
    int lb = 0;
    int ub = 0;
    std::vector<int> order_off;
    std::vector<int> order_item;
    std::vector<int> order_qty;
    std::vector<int> order_size;
    std::vector<int> aisle_off;
    std::vector<int> aisle_item;
    std::vector<int> aisle_qty;
    std::vector<int> total_demand;  // aggregate of all orders, per item
    std::vector<int> stock_total;   // aggregate of all aisles, per item
};

struct Solution {
    std::vector<int> selected_orders;
    std::vector<int> visited_aisles;
    double objective = 0.0;
    std::vector<int> demand;
    int total_units = 0;
};

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "aisle_grasp_solver: " << msg << "\n";
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    bool got_alpha = false, got_scoring = false, got_aisle_score = false;
    bool got_packing = false, got_iters = false, got_greedy = false;
    bool got_ls_aisle = false, got_ls_order = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) {
            if (!opt.instance_path.empty()) die("multiple positional args");
            opt.instance_path = a;
            continue;
        }
        auto eq = a.find('=');
        std::string key = a.substr(2, eq == std::string::npos ? std::string::npos : eq - 2);
        std::string val = eq == std::string::npos ? "" : a.substr(eq + 1);
        if (key == "alpha") { opt.alpha = std::stod(val); got_alpha = true; }
        else if (key == "scoring") {
            if (val == "static") opt.scoring = Scoring::Static;
            else if (val == "adaptive") opt.scoring = Scoring::Adaptive;
            else die("invalid scoring: " + val);
            got_scoring = true;
        }
        else if (key == "aisle-score") {
            if (val == "useful") opt.aisle_score = AisleScore::Useful;
            else if (val == "units") opt.aisle_score = AisleScore::Units;
            else if (val == "variety") opt.aisle_score = AisleScore::Variety;
            else if (val == "mixed") opt.aisle_score = AisleScore::Mixed;
            else die("invalid aisle-score: " + val);
            got_aisle_score = true;
        }
        else if (key == "packing-order") {
            if (val == "asc") opt.packing_order = PackOrder::Asc;
            else if (val == "desc") opt.packing_order = PackOrder::Desc;
            else if (val == "shuffle") opt.packing_order = PackOrder::Shuffle;
            else die("invalid packing-order: " + val);
            got_packing = true;
        }
        else if (key == "max-iterations") { opt.max_iterations = std::stoi(val); got_iters = true; }
        else if (key == "greedy") {
            if (val == "simple") opt.greedy = Greedy::Simple;
            else if (val == "multi") opt.greedy = Greedy::Multi;
            else die("invalid greedy: " + val);
            got_greedy = true;
        }
        else if (key == "local-search-aisle") {
            if (val == "none") opt.local_search_aisle = LocalSearch::None;
            else if (val == "swap") opt.local_search_aisle = LocalSearch::Swap;
            else if (val == "full") opt.local_search_aisle = LocalSearch::Full;
            else die("invalid local-search-aisle: " + val);
            got_ls_aisle = true;
        }
        else if (key == "local-search-order") {
            if (val == "none") opt.local_search_order = LocalSearch::None;
            else if (val == "swap") opt.local_search_order = LocalSearch::Swap;
            else if (val == "full") opt.local_search_order = LocalSearch::Full;
            else die("invalid local-search-order: " + val);
            got_ls_order = true;
        }
        else if (key == "seed") { opt.seed = static_cast<uint64_t>(std::stoll(val)); opt.seed_set = true; }
        else if (key == "time-limit") opt.time_limit = std::stod(val);
        else die("unknown option: " + key);
    }
    if (opt.instance_path.empty()) die("missing instance path");
    if (!got_alpha || !got_scoring || !got_aisle_score || !got_packing
        || !got_iters || !got_greedy || !got_ls_aisle || !got_ls_order)
        die("missing required option");
    if (opt.alpha < 0.0 || opt.alpha > 1.0) die("alpha out of range");
    if (opt.max_iterations <= 0) die("max-iterations must be positive");
    return opt;
}

Problem load_instance(const std::string& path) {
    std::ifstream f(path);
    if (!f) die("cannot open instance: " + path);
    Problem p;
    f >> p.n_orders >> p.n_items >> p.n_aisles;
    if (!f) die("malformed header");

    p.order_off.assign(p.n_orders + 1, 0);
    p.order_size.assign(p.n_orders, 0);
    p.total_demand.assign(p.n_items, 0);
    p.order_item.reserve(p.n_orders * 4);
    p.order_qty.reserve(p.n_orders * 4);
    for (int o = 0; o < p.n_orders; ++o) {
        int k; f >> k;
        p.order_off[o + 1] = p.order_off[o] + k;
        int s = 0;
        for (int j = 0; j < k; ++j) {
            int it, q; f >> it >> q;
            p.order_item.push_back(it);
            p.order_qty.push_back(q);
            p.total_demand[it] += q;
            s += q;
        }
        p.order_size[o] = s;
    }

    p.aisle_off.assign(p.n_aisles + 1, 0);
    p.stock_total.assign(p.n_items, 0);
    p.aisle_item.reserve(p.n_aisles * 4);
    p.aisle_qty.reserve(p.n_aisles * 4);
    for (int a = 0; a < p.n_aisles; ++a) {
        int k; f >> k;
        p.aisle_off[a + 1] = p.aisle_off[a] + k;
        for (int j = 0; j < k; ++j) {
            int it, q; f >> it >> q;
            p.aisle_item.push_back(it);
            p.aisle_qty.push_back(q);
            p.stock_total[it] += q;
        }
    }
    f >> p.lb >> p.ub;
    return p;
}

// ---- Emission ----------------------------------------------------------

void emit(const Solution& s) {
    std::ostringstream out;
    out << std::setprecision(17);
    out << "{\"objective\":" << s.objective << ",\"selected_orders\":[";
    for (size_t i = 0; i < s.selected_orders.size(); ++i) {
        if (i) out << ',';
        out << s.selected_orders[i];
    }
    out << "],\"visited_aisles\":[";
    for (size_t i = 0; i < s.visited_aisles.size(); ++i) {
        if (i) out << ',';
        out << s.visited_aisles[i];
    }
    out << "]}\n";
    std::cout << out.str();
    std::cout.flush();
}

// ---- Aisle selection (for order-level LS) -----------------------------

std::vector<int> simple_greedy_aisles(const Problem& p, std::vector<int> demand) {
    std::vector<int> order(p.n_aisles);
    std::vector<int> score(p.n_aisles, 0);
    for (int a = 0; a < p.n_aisles; ++a) {
        int s = 0;
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int d = demand[p.aisle_item[j]];
            if (d > 0) s += std::min(d, p.aisle_qty[j]);
        }
        score[a] = s;
        order[a] = a;
    }
    std::sort(order.begin(), order.end(),
              [&](int x, int y) { return score[x] > score[y]; });
    int remaining = 0;
    for (int v : demand) if (v > 0) remaining += v;
    std::vector<int> selected;
    selected.reserve(16);
    for (int a : order) {
        if (remaining == 0) break;
        int real = 0;
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int d = demand[p.aisle_item[j]];
            if (d > 0) real += std::min(d, p.aisle_qty[j]);
        }
        if (real == 0) continue;
        selected.push_back(a);
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
            int it = p.aisle_item[j];
            int d = demand[it];
            if (d > 0) {
                int take = std::min(d, p.aisle_qty[j]);
                demand[it] = d - take;
                remaining -= take;
            }
        }
    }
    return selected;
}

std::vector<int> multi_greedy_aisles(const Problem& p, std::vector<int> demand) {
    std::vector<char> used(p.n_aisles, 0);
    std::vector<int> selected;
    int remaining = 0;
    for (int v : demand) if (v > 0) remaining += v;
    while (remaining > 0) {
        int best = -1, best_score = 0;
        for (int a = 0; a < p.n_aisles; ++a) {
            if (used[a]) continue;
            int s = 0;
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
                int d = demand[p.aisle_item[j]];
                if (d > 0) s += std::min(d, p.aisle_qty[j]);
            }
            if (s > best_score) { best_score = s; best = a; }
        }
        if (best_score == 0) break;
        used[best] = 1;
        selected.push_back(best);
        for (int j = p.aisle_off[best]; j < p.aisle_off[best + 1]; ++j) {
            int it = p.aisle_item[j];
            int d = demand[it];
            if (d > 0) {
                int take = std::min(d, p.aisle_qty[j]);
                demand[it] = d - take;
                remaining -= take;
            }
        }
    }
    return selected;
}

std::vector<int> select_aisles(const Options& opt, const Problem& p,
                               const std::vector<int>& demand) {
    if (opt.greedy == Greedy::Multi) return multi_greedy_aisles(p, demand);
    return simple_greedy_aisles(p, demand);
}

// ---- Bin packing orders against an inventory --------------------------

void pack_orders(const Problem& p, const std::vector<int>& sequence,
                 const std::vector<int>& inventory, int ub,
                 std::vector<int>& scratch,
                 std::vector<int>& selected_out, int& total_out) {
    scratch = inventory;
    selected_out.clear();
    total_out = 0;
    for (int idx : sequence) {
        int size = p.order_size[idx];
        if (total_out + size > ub) continue;
        bool ok = true;
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
            if (scratch[p.order_item[j]] < p.order_qty[j]) { ok = false; break; }
        }
        if (!ok) continue;
        selected_out.push_back(idx);
        total_out += size;
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
            scratch[p.order_item[j]] -= p.order_qty[j];
        }
    }
}

// ---- Aisle scoring -----------------------------------------------------

std::vector<double> compute_static_scores(const Options& opt, const Problem& p) {
    std::vector<double> s(p.n_aisles, 0.0);
    for (int a = 0; a < p.n_aisles; ++a) {
        double val = 0.0;
        if (opt.aisle_score == AisleScore::Useful) {
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
                int it = p.aisle_item[j];
                int gap = p.total_demand[it];
                if (gap > 0) val += std::min(p.aisle_qty[j], gap);
            }
        } else if (opt.aisle_score == AisleScore::Units) {
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j)
                val += p.aisle_qty[j];
        } else if (opt.aisle_score == AisleScore::Variety) {
            val = static_cast<double>(p.aisle_off[a + 1] - p.aisle_off[a]);
        } else {  // Mixed: units * variety
            int units = 0;
            for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j)
                units += p.aisle_qty[j];
            int variety = p.aisle_off[a + 1] - p.aisle_off[a];
            val = static_cast<double>(units) * static_cast<double>(variety);
        }
        s[a] = val;
    }
    return s;
}

// "useful" recomputed against the current unmet-demand gap; other scores
// don't depend on inventory and the caller falls back to the static value.
double adaptive_useful_score(const Problem& p, int a,
                             const std::vector<int>& inventory) {
    double val = 0.0;
    for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j) {
        int it = p.aisle_item[j];
        int gap = p.total_demand[it] - inventory[it];
        if (gap > 0) val += std::min(p.aisle_qty[j], gap);
    }
    return val;
}

// ---- Packing sequence --------------------------------------------------

std::vector<int> build_packing_sequence(const Options& opt, const Problem& p,
                                        std::mt19937& rng) {
    std::vector<int> seq(p.n_orders);
    std::iota(seq.begin(), seq.end(), 0);
    if (opt.packing_order == PackOrder::Shuffle) {
        std::shuffle(seq.begin(), seq.end(), rng);
    } else {
        const bool desc = (opt.packing_order == PackOrder::Desc);
        std::stable_sort(seq.begin(), seq.end(), [&](int x, int y) {
            return desc ? p.order_size[x] > p.order_size[y]
                        : p.order_size[x] < p.order_size[y];
        });
    }
    return seq;
}

// ---- Helpers -----------------------------------------------------------

void compute_demand(const Problem& p, const std::vector<int>& selected,
                    std::vector<int>& demand) {
    std::fill(demand.begin(), demand.end(), 0);
    for (int idx : selected) {
        for (int j = p.order_off[idx]; j < p.order_off[idx + 1]; ++j) {
            demand[p.order_item[j]] += p.order_qty[j];
        }
    }
}

void build_inventory(const Problem& p, const std::vector<int>& visited,
                     std::vector<int>& inv) {
    std::fill(inv.begin(), inv.end(), 0);
    for (int a : visited) {
        for (int j = p.aisle_off[a]; j < p.aisle_off[a + 1]; ++j)
            inv[p.aisle_item[j]] += p.aisle_qty[j];
    }
}

// ---- Construction ------------------------------------------------------

bool construct(const Options& opt, const Problem& p, std::mt19937& rng,
               const std::vector<double>& static_scores,
               const std::vector<int>& packing_seq,
               Solution& out) {
    std::vector<int> inventory(p.n_items, 0);
    std::vector<int> remaining(p.n_aisles);
    std::iota(remaining.begin(), remaining.end(), 0);
    std::vector<int> ordered;
    ordered.reserve(p.n_aisles);

    std::vector<double> scores;
    std::vector<int> rcl;
    scores.reserve(p.n_aisles);
    rcl.reserve(p.n_aisles);

    std::vector<int> pack_scratch(p.n_items, 0);
    std::vector<int> pack_selected;
    pack_selected.reserve(p.n_orders);

    int best_units = 0;
    double best_obj = 0.0;
    std::vector<int> best_orders;
    std::vector<int> best_aisles;

    const bool adaptive_useful = (opt.scoring == Scoring::Adaptive
                                  && opt.aisle_score == AisleScore::Useful);

    while (!remaining.empty()) {
        scores.assign(remaining.size(), 0.0);
        if (adaptive_useful) {
            for (size_t k = 0; k < remaining.size(); ++k)
                scores[k] = adaptive_useful_score(p, remaining[k], inventory);
        } else {
            for (size_t k = 0; k < remaining.size(); ++k)
                scores[k] = static_scores[remaining[k]];
        }

        double g_max = scores[0], g_min = scores[0];
        for (double s : scores) { if (s > g_max) g_max = s; if (s < g_min) g_min = s; }
        if (g_max <= 0.0) break;
        double threshold = g_max - opt.alpha * (g_max - g_min);
        rcl.clear();
        for (size_t k = 0; k < remaining.size(); ++k)
            if (scores[k] >= threshold) rcl.push_back(remaining[k]);
        std::uniform_int_distribution<size_t> dist(0, rcl.size() - 1);
        int pick = rcl[dist(rng)];

        ordered.push_back(pick);
        for (int j = p.aisle_off[pick]; j < p.aisle_off[pick + 1]; ++j)
            inventory[p.aisle_item[j]] += p.aisle_qty[j];
        for (size_t k = 0; k < remaining.size(); ++k) {
            if (remaining[k] == pick) {
                remaining[k] = remaining.back();
                remaining.pop_back();
                break;
            }
        }

        int k_used = static_cast<int>(ordered.size());
        if (static_cast<double>(p.ub) / k_used <= best_obj) break;

        int total = 0;
        pack_orders(p, packing_seq, inventory, p.ub, pack_scratch,
                    pack_selected, total);
        if (total < p.lb) continue;
        double obj = static_cast<double>(total) / k_used;
        if (obj > best_obj) {
            best_obj = obj;
            best_units = total;
            best_orders = pack_selected;
            best_aisles = ordered;
        }
    }

    if (best_orders.empty()) return false;
    out.selected_orders = std::move(best_orders);
    out.visited_aisles = std::move(best_aisles);
    out.total_units = best_units;
    out.objective = best_obj;
    out.demand.assign(p.n_items, 0);
    compute_demand(p, out.selected_orders, out.demand);
    return true;
}

// ---- Aisle-level local search -----------------------------------------
//
// Swap/drop/add moves on the visited-aisles set. pack_orders is called on
// each candidate inventory to refresh the selected orders; accept on first
// improvement of the ratio objective.

bool try_aisle_swap(const Problem& p, const std::vector<int>& packing_seq,
                    Solution& cur) {
    int k = static_cast<int>(cur.visited_aisles.size());
    std::vector<char> in_vis(p.n_aisles, 0);
    for (int a : cur.visited_aisles) in_vis[a] = 1;

    std::vector<int> inv(p.n_items, 0);
    build_inventory(p, cur.visited_aisles, inv);

    std::vector<int> pack_scratch(p.n_items, 0);
    std::vector<int> pack_selected;
    pack_selected.reserve(p.n_orders);

    for (size_t vi = 0; vi < cur.visited_aisles.size(); ++vi) {
        int v = cur.visited_aisles[vi];
        for (int j = p.aisle_off[v]; j < p.aisle_off[v + 1]; ++j)
            inv[p.aisle_item[j]] -= p.aisle_qty[j];

        for (int u = 0; u < p.n_aisles; ++u) {
            if (in_vis[u]) continue;
            for (int j = p.aisle_off[u]; j < p.aisle_off[u + 1]; ++j)
                inv[p.aisle_item[j]] += p.aisle_qty[j];

            int total = 0;
            pack_orders(p, packing_seq, inv, p.ub, pack_scratch,
                        pack_selected, total);
            if (total >= p.lb) {
                double new_obj = static_cast<double>(total) / k;
                if (new_obj > cur.objective) {
                    std::vector<int> new_visited;
                    new_visited.reserve(cur.visited_aisles.size());
                    for (int a : cur.visited_aisles)
                        if (a != v) new_visited.push_back(a);
                    new_visited.push_back(u);
                    cur.selected_orders = std::move(pack_selected);
                    cur.visited_aisles = std::move(new_visited);
                    cur.total_units = total;
                    cur.objective = new_obj;
                    compute_demand(p, cur.selected_orders, cur.demand);
                    return true;
                }
            }
            for (int j = p.aisle_off[u]; j < p.aisle_off[u + 1]; ++j)
                inv[p.aisle_item[j]] -= p.aisle_qty[j];
        }

        for (int j = p.aisle_off[v]; j < p.aisle_off[v + 1]; ++j)
            inv[p.aisle_item[j]] += p.aisle_qty[j];
    }
    return false;
}

bool try_aisle_drop(const Problem& p, const std::vector<int>& packing_seq,
                    Solution& cur) {
    int k = static_cast<int>(cur.visited_aisles.size());
    if (k <= 1) return false;

    std::vector<int> inv(p.n_items, 0);
    build_inventory(p, cur.visited_aisles, inv);

    std::vector<int> pack_scratch(p.n_items, 0);
    std::vector<int> pack_selected;
    pack_selected.reserve(p.n_orders);

    for (size_t vi = 0; vi < cur.visited_aisles.size(); ++vi) {
        int v = cur.visited_aisles[vi];
        for (int j = p.aisle_off[v]; j < p.aisle_off[v + 1]; ++j)
            inv[p.aisle_item[j]] -= p.aisle_qty[j];

        int total = 0;
        pack_orders(p, packing_seq, inv, p.ub, pack_scratch,
                    pack_selected, total);
        if (total >= p.lb) {
            int new_k = k - 1;
            double new_obj = static_cast<double>(total) / new_k;
            if (new_obj > cur.objective) {
                std::vector<int> new_visited;
                new_visited.reserve(cur.visited_aisles.size() - 1);
                for (int a : cur.visited_aisles)
                    if (a != v) new_visited.push_back(a);
                cur.selected_orders = std::move(pack_selected);
                cur.visited_aisles = std::move(new_visited);
                cur.total_units = total;
                cur.objective = new_obj;
                compute_demand(p, cur.selected_orders, cur.demand);
                return true;
            }
        }

        for (int j = p.aisle_off[v]; j < p.aisle_off[v + 1]; ++j)
            inv[p.aisle_item[j]] += p.aisle_qty[j];
    }
    return false;
}

bool try_aisle_add(const Problem& p, const std::vector<int>& packing_seq,
                   Solution& cur) {
    int k = static_cast<int>(cur.visited_aisles.size());
    std::vector<char> in_vis(p.n_aisles, 0);
    for (int a : cur.visited_aisles) in_vis[a] = 1;

    std::vector<int> inv(p.n_items, 0);
    build_inventory(p, cur.visited_aisles, inv);

    std::vector<int> pack_scratch(p.n_items, 0);
    std::vector<int> pack_selected;
    pack_selected.reserve(p.n_orders);

    for (int u = 0; u < p.n_aisles; ++u) {
        if (in_vis[u]) continue;
        for (int j = p.aisle_off[u]; j < p.aisle_off[u + 1]; ++j)
            inv[p.aisle_item[j]] += p.aisle_qty[j];

        int total = 0;
        pack_orders(p, packing_seq, inv, p.ub, pack_scratch,
                    pack_selected, total);
        if (total >= p.lb) {
            int new_k = k + 1;
            double new_obj = static_cast<double>(total) / new_k;
            if (new_obj > cur.objective) {
                std::vector<int> new_visited = cur.visited_aisles;
                new_visited.push_back(u);
                cur.selected_orders = std::move(pack_selected);
                cur.visited_aisles = std::move(new_visited);
                cur.total_units = total;
                cur.objective = new_obj;
                compute_demand(p, cur.selected_orders, cur.demand);
                return true;
            }
        }

        for (int j = p.aisle_off[u]; j < p.aisle_off[u + 1]; ++j)
            inv[p.aisle_item[j]] -= p.aisle_qty[j];
    }
    return false;
}

void improve_aisles(const Options& opt, const Problem& p,
                    const std::vector<int>& packing_seq, Solution& cur) {
    bool improved = true;
    while (improved) {
        improved = false;
        if (try_aisle_swap(p, packing_seq, cur)) { improved = true; continue; }
        if (opt.local_search_aisle == LocalSearch::Full) {
            if (try_aisle_drop(p, packing_seq, cur)) { improved = true; continue; }
            if (try_aisle_add(p, packing_seq, cur))  { improved = true; continue; }
        }
    }
}

// ---- Order-level local search -----------------------------------------
//
// Swap/drop/add moves on the selected-orders set; visited_aisles is
// recomputed via select_aisles on the new demand.

bool demand_within_stock(const std::vector<int>& demand,
                         const std::vector<int>& stock_total) {
    for (size_t i = 0; i < demand.size(); ++i)
        if (demand[i] > stock_total[i]) return false;
    return true;
}

bool try_order_swap(const Options& opt, const Problem& p, Solution& cur) {
    std::vector<char> in_sel(p.n_orders, 0);
    for (int i : cur.selected_orders) in_sel[i] = 1;

    for (size_t si = 0; si < cur.selected_orders.size(); ++si) {
        int s_idx = cur.selected_orders[si];
        int s_size = p.order_size[s_idx];
        std::vector<int> demand_minus = cur.demand;
        for (int j = p.order_off[s_idx]; j < p.order_off[s_idx + 1]; ++j)
            demand_minus[p.order_item[j]] -= p.order_qty[j];

        for (int u = 0; u < p.n_orders; ++u) {
            if (in_sel[u]) continue;
            int u_size = p.order_size[u];
            if (u_size == 0) continue;
            int new_total = cur.total_units - s_size + u_size;
            if (new_total > p.ub || new_total < p.lb) continue;

            for (int j = p.order_off[u]; j < p.order_off[u + 1]; ++j)
                demand_minus[p.order_item[j]] += p.order_qty[j];

            if (demand_within_stock(demand_minus, p.stock_total)) {
                auto new_visited = select_aisles(opt, p, demand_minus);
                if (!new_visited.empty()) {
                    double new_obj = static_cast<double>(new_total) / new_visited.size();
                    if (new_obj > cur.objective) {
                        cur.selected_orders[si] = u;
                        cur.total_units = new_total;
                        cur.demand = std::move(demand_minus);
                        cur.visited_aisles = std::move(new_visited);
                        cur.objective = new_obj;
                        return true;
                    }
                }
            }

            for (int j = p.order_off[u]; j < p.order_off[u + 1]; ++j)
                demand_minus[p.order_item[j]] -= p.order_qty[j];
        }
    }
    return false;
}

bool try_order_drop(const Options& opt, const Problem& p, Solution& cur) {
    if (cur.selected_orders.size() <= 1) return false;
    for (size_t si = 0; si < cur.selected_orders.size(); ++si) {
        int s_idx = cur.selected_orders[si];
        int s_size = p.order_size[s_idx];
        int new_total = cur.total_units - s_size;
        if (new_total < p.lb) continue;

        std::vector<int> new_demand = cur.demand;
        for (int j = p.order_off[s_idx]; j < p.order_off[s_idx + 1]; ++j)
            new_demand[p.order_item[j]] -= p.order_qty[j];

        auto new_visited = select_aisles(opt, p, new_demand);
        if (new_visited.empty()) continue;
        double new_obj = static_cast<double>(new_total) / new_visited.size();
        if (new_obj > cur.objective) {
            cur.selected_orders.erase(cur.selected_orders.begin() + si);
            cur.total_units = new_total;
            cur.demand = std::move(new_demand);
            cur.visited_aisles = std::move(new_visited);
            cur.objective = new_obj;
            return true;
        }
    }
    return false;
}

bool try_order_add(const Options& opt, const Problem& p, Solution& cur) {
    std::vector<char> in_sel(p.n_orders, 0);
    for (int i : cur.selected_orders) in_sel[i] = 1;
    for (int u = 0; u < p.n_orders; ++u) {
        if (in_sel[u]) continue;
        int u_size = p.order_size[u];
        if (u_size == 0) continue;
        int new_total = cur.total_units + u_size;
        if (new_total > p.ub) continue;

        std::vector<int> new_demand = cur.demand;
        for (int j = p.order_off[u]; j < p.order_off[u + 1]; ++j)
            new_demand[p.order_item[j]] += p.order_qty[j];

        if (!demand_within_stock(new_demand, p.stock_total)) continue;

        auto new_visited = select_aisles(opt, p, new_demand);
        if (new_visited.empty()) continue;
        double new_obj = static_cast<double>(new_total) / new_visited.size();
        if (new_obj > cur.objective) {
            cur.selected_orders.push_back(u);
            cur.total_units = new_total;
            cur.demand = std::move(new_demand);
            cur.visited_aisles = std::move(new_visited);
            cur.objective = new_obj;
            return true;
        }
    }
    return false;
}

void improve_orders(const Options& opt, const Problem& p, Solution& cur) {
    bool improved = true;
    while (improved) {
        improved = false;
        if (try_order_swap(opt, p, cur)) { improved = true; continue; }
        if (opt.local_search_order == LocalSearch::Full) {
            if (try_order_drop(opt, p, cur)) { improved = true; continue; }
            if (try_order_add(opt, p, cur))  { improved = true; continue; }
        }
    }
}

// ---- Main --------------------------------------------------------------

int run(const Options& opt) {
    Problem p = load_instance(opt.instance_path);
    Solution best;
    best.objective = 0.0;
    emit(best);
    if (p.n_orders == 0 || p.n_aisles == 0) {
        emit(best);
        return 0;
    }

    std::mt19937 rng;
    if (opt.seed_set) rng.seed(static_cast<uint32_t>(opt.seed));
    else rng.seed(std::random_device{}());

    auto static_scores = compute_static_scores(opt, p);

    auto t_start = Clock::now();
    auto time_expired = [&]() {
        if (opt.time_limit <= 0) return false;
        std::chrono::duration<double> d = Clock::now() - t_start;
        return d.count() >= opt.time_limit;
    };

    for (int it = 0; it < opt.max_iterations; ++it) {
        if (time_expired()) break;
        auto packing_seq = build_packing_sequence(opt, p, rng);
        Solution built;
        if (!construct(opt, p, rng, static_scores, packing_seq, built)) continue;

        if (built.objective > best.objective) {
            best = built;
            emit(best);
        }

        if (opt.local_search_aisle != LocalSearch::None) {
            if (time_expired()) break;
            improve_aisles(opt, p, packing_seq, built);
        }

        if (opt.local_search_order != LocalSearch::None) {
            if (time_expired()) break;
            improve_orders(opt, p, built);
        }

        if (built.objective > best.objective) {
            best = built;
            emit(best);
        }
    }

    emit(best);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "aisle_grasp_solver exception: " << e.what() << "\n";
        return 3;
    }
}
