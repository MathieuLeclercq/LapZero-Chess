#include "mcts.hpp"
#include "mcts_reservation.hpp"
#include "test_support.hpp"

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

class Gate {
public:
    explicit Gate(int participants) : m_remaining(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        --m_remaining;
        if (m_remaining == 0) {
            m_cv.notify_all();
            return;
        }
        m_cv.wait(lock, [&]() { return m_remaining == 0; });
    }

private:
    int m_remaining;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

void test_move_and_exception_release_everything() {
    MCTSNode root(0.0f);
    MCTSNode child(1.0f, 8, &root);
    root.visit_count = 4;
    root.total_value = 1.0f;
    const float q_before = root.q_value();

    try {
        PathReservation first;
        first.reserve(&root);
        first.reserve(&child);
        require_test(first.try_claim(&child), "claim failed");
        PathReservation moved(std::move(first));
        require_test(root.n_in_flight.load() == 1,
                     "move duplicated root reservation");
        require_test(child.n_in_flight.load() == 1,
                     "move duplicated child reservation");
        require_test(child.state.load() == NodeState::Pending,
                     "claim did not publish Pending");
        require_test(root.q_value() == q_before,
                     "reservation changed Q");
        throw std::runtime_error("test exception");
    }
    catch (const std::runtime_error&) {
    }

    require_test(root.n_in_flight.load() == 0,
                 "root reservation leaked after exception");
    require_test(child.n_in_flight.load() == 0,
                 "child reservation leaked after exception");
    require_test(child.state.load() == NodeState::Unexpanded,
                 "unpublished Pending leaked after exception");
    require_test(root.q_value() == q_before,
                 "release changed Q");
}

void test_publish_keeps_state_and_releases_path() {
    MCTSNode root(0.0f);
    {
        PathReservation reservation;
        reservation.reserve(&root);
        require_test(reservation.try_claim(&root), "claim failed");
        reservation.publish(NodeState::Expanded);
    }

    require_test(root.n_in_flight.load() == 0,
                 "published reservation leaked in-flight count");
    require_test(root.state.load() == NodeState::Expanded,
                 "published state was rolled back");
}

void test_move_assignment_releases_previous_reservation() {
    MCTSNode old_root(0.0f);
    MCTSNode new_root(0.0f);
    PathReservation destination;
    destination.reserve(&old_root);
    require_test(destination.try_claim(&old_root), "old claim failed");

    PathReservation source;
    source.reserve(&new_root);
    require_test(source.try_claim(&new_root), "new claim failed");
    destination = std::move(source);

    require_test(old_root.n_in_flight.load() == 0,
                 "move assignment leaked previous path");
    require_test(old_root.state.load() == NodeState::Unexpanded,
                 "move assignment leaked previous Pending");
    require_test(new_root.n_in_flight.load() == 1,
                 "move assignment lost new path");
}

void test_only_one_concurrent_guard_owns_pending() {
    constexpr int THREADS = 8;
    MCTSNode root(0.0f);
    Gate start(THREADS);
    Gate attempted(THREADS);
    std::atomic<int> winners{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < THREADS; ++i) {
        threads.emplace_back([&]() {
            PathReservation reservation;
            reservation.reserve(&root);
            start.wait();
            const bool won = reservation.try_claim(&root);
            if (won) {
                winners.fetch_add(1, std::memory_order_relaxed);
            }
            attempted.wait();
            if (won) {
                reservation.publish(NodeState::Expanded);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    require_test(winners.load() == 1, "Pending had multiple owners");
    require_test(root.n_in_flight.load() == 0,
                 "concurrent reservations leaked");
    require_test(root.state.load() == NodeState::Expanded,
                 "loser rolled back winner publication");
}

}  // namespace

int main() {
    try {
        test_move_and_exception_release_everything();
        test_publish_keeps_state_and_releases_path();
        test_move_assignment_releases_previous_reservation();
        test_only_one_concurrent_guard_owns_pending();
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
