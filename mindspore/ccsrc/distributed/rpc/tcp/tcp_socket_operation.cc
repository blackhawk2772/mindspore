/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "distributed/rpc/tcp/tcp_socket_operation.h"

namespace mindspore {
namespace distributed {
namespace rpc {
constexpr int EAGAIN_RETRY = 1024000;

ssize_t TCPSocketOperation::ReceivePeek(Connection *connection, char *recvBuf, uint32_t recvLen) {
  return recv(connection->socket_fd, recvBuf, recvLen, MSG_PEEK);
}

int TCPSocketOperation::Receive(Connection *connection, char *recvBuf, size_t totalRecvLen, size_t *recvLen) {
  if (connection == nullptr || recvBuf == nullptr || recvLen == nullptr) {
    return IO_RW_ERROR;
  }
  char *curRecvBuf = recvBuf;
  int fd = connection->socket_fd;

  *recvLen = 0;
  while (*recvLen != totalRecvLen) {
    ssize_t retval = recv(fd, curRecvBuf, totalRecvLen - *recvLen, static_cast<int>(0));
    if (retval > 0) {
      *recvLen += retval;
      if (*recvLen == totalRecvLen) {
        return IO_RW_OK;
      }
      curRecvBuf = curRecvBuf + retval;
      // Failed to receive message.
    } else if (retval < 0) {
      if (EAGAIN == errno) {
        return IO_RW_OK;
      } else if (ECONNRESET == errno || ECONNABORTED == errno || ENOTCONN == errno || EPIPE == errno) {
        connection->error_code = errno;
        return IO_RW_ERROR;
      } else {
        return IO_RW_OK;
      }
    } else {
      connection->error_code = errno;
      return IO_RW_ERROR;
    }
  }
  return IO_RW_OK;
}

int TCPSocketOperation::ReceiveMessage(Connection *connection, struct msghdr *recvMsg, size_t totalRecvLen,
                                       size_t *recvLen) {
  if (totalRecvLen == 0) {
    return IO_RW_OK;
  }

  while (*recvLen < totalRecvLen) {
    auto retval = recvmsg(connection->socket_fd, recvMsg, 0);
    if (retval > 0) {
      *recvLen += retval;
      if (*recvLen == totalRecvLen) {
        recvMsg->msg_iovlen = 0;
        break;
      }

      unsigned int iovlen = recvMsg->msg_iovlen;
      if (iovlen > 0) {
        size_t tmpLen = 0;
        for (unsigned int i = 0; i < iovlen; ++i) {
          if (recvMsg->msg_iov[i].iov_len + tmpLen <= static_cast<size_t>(retval)) {
            tmpLen += recvMsg->msg_iov[i].iov_len;
          } else {
            recvMsg->msg_iov[i].iov_len -= IntToSize(retval - tmpLen);
            recvMsg->msg_iov[i].iov_base =
              reinterpret_cast<char *>(recvMsg->msg_iov[i].iov_base) + static_cast<unsigned int>(retval) - tmpLen;

            recvMsg->msg_iov = &recvMsg->msg_iov[i];
            recvMsg->msg_iovlen -= i;
            break;
          }
        }
      }
    } else if (retval == 0) {
      return IO_RW_ERROR;
    } else {
      if (EAGAIN == errno) {
        return IO_RW_OK;
      } else if (ECONNRESET == errno || ECONNABORTED == errno || ENOTCONN == errno || EPIPE == errno) {
        connection->error_code = errno;
        return IO_RW_ERROR;
      } else {
        return IO_RW_OK;
      }
    }
  }
  return IO_RW_OK;
}

int TCPSocketOperation::SendMessage(Connection *connection, struct msghdr *sendMsg, size_t totalSendLen,
                                    size_t *sendLen) {
  if (connection == nullptr || sendMsg == nullptr || sendLen == nullptr) {
    return IO_RW_ERROR;
  }
  int eagainCount = 0;
  // Print retry log interval.
  const int print_interval = 10000;
  const int sleep_interval_factor = 10;
  *sendLen = 0;

  while (*sendLen != totalSendLen) {
    auto retval = sendmsg(connection->socket_fd, sendMsg, MSG_NOSIGNAL);
    if (retval < 0) {
      ++eagainCount;
      if (errno != EAGAIN) {
        MS_LOG(ERROR) << "Failed to call sendmsg and errno is: " << errno;
        connection->error_code = errno;
        return IO_RW_ERROR;
      } else if (eagainCount == EAGAIN_RETRY) {
        MS_LOG(ERROR) << "Failed to call sendmsg after retry " + std::to_string(EAGAIN_RETRY) + " times and errno is: "
                      << errno;
        *sendLen = 0;
        return IO_RW_OK;
      }
      if (eagainCount % print_interval == 0) {
        MS_LOG(WARNING) << "Retry(" + std::to_string(eagainCount) + "/" + std::to_string(EAGAIN_RETRY) +
                             ") sending ...";
      }
      std::this_thread::sleep_for(eagainCount * std::chrono::microseconds(sleep_interval_factor));
    } else {
      *sendLen += retval;

      if (*sendLen == totalSendLen) {
        sendMsg->msg_iovlen = 0;
        break;
      }

      size_t tmpBytes = 0;
      for (unsigned int i = 0; i < sendMsg->msg_iovlen; ++i) {
        if (sendMsg->msg_iov[i].iov_len + tmpBytes < static_cast<size_t>(retval)) {
          tmpBytes += sendMsg->msg_iov[i].iov_len;
        } else {
          sendMsg->msg_iov[i].iov_len -= (retval - tmpBytes);
          sendMsg->msg_iov[i].iov_base =
            reinterpret_cast<char *>(sendMsg->msg_iov[i].iov_base) + static_cast<unsigned int>(retval) - tmpBytes;

          sendMsg->msg_iov = &sendMsg->msg_iov[i];
          sendMsg->msg_iovlen -= i;
          break;
        }
      }
      eagainCount = 0;
    }
  }
  return IO_RW_OK;
}

void TCPSocketOperation::Close(Connection *connection) {
  if (connection == nullptr) {
    return;
  }
  (void)close(connection->socket_fd);
  connection->socket_fd = -1;
}

// accept new conn event handle
void TCPSocketOperation::NewConnEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  if (conn == nullptr) {
    return;
  }
  conn->state = ConnectionState::kConnected;
  return;
}

void TCPSocketOperation::ConnEstablishedEventHandler(int fd, uint32_t events, void *context) {
  Connection *conn = reinterpret_cast<Connection *>(context);
  if (conn == nullptr) {
    return;
  }
  conn->state = ConnectionState::kConnected;
  return;
}
}  // namespace rpc
}  // namespace distributed
}  // namespace mindspore
