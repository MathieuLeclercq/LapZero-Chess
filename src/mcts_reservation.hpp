#pragma once

#include <cstdint>
#include <vector>

struct MCTSNode;

enum class NodeState : std::uint8_t {
    Unexpanded,
    Pending,
    Expanded,
    Terminal,
};

class PathReservation {
public:
    PathReservation() = default;
    PathReservation(PathReservation&& other) noexcept;
    PathReservation& operator=(PathReservation&& other) noexcept;
    PathReservation(const PathReservation&) = delete;
    PathReservation& operator=(const PathReservation&) = delete;
    ~PathReservation();

    void reserve(MCTSNode* node, std::uint32_t units = 1);
    bool try_claim(MCTSNode* node);
    void publish(NodeState state) noexcept;
    void release() noexcept;

private:
    std::vector<MCTSNode*> m_path;
    MCTSNode* m_owned_pending = nullptr;
};
