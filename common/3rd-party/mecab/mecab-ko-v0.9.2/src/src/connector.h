// MeCab -- Yet Another Part-of-Speech and Morphological Analyzer
//
//
//  Copyright(C) 2001-2006 Taku Kudo <taku@chasen.org>
//  Copyright(C) 2004-2006 Nippon Telegraph and Telephone Corporation
#ifndef MECAB_CONNECTOR_H_
#define MECAB_CONNECTOR_H_

#include "mecab.h"
#include "mmap.h"
#include "common.h"
#include "scoped_ptr.h"

namespace MeCab {
class Param;

class Connector {
 private:
  scoped_ptr<Mmap<short> >  cmmap_;
  short          *matrix_;
  unsigned short  lsize_;
  unsigned short  rsize_;
  whatlog         what_;

  // mecab-ko
  class SpacePenalty {
    public:
      unsigned short posid_;
      int penalty_cost_;
      SpacePenalty(unsigned short posid, int penalty_cost)
        : posid_(posid)
        , penalty_cost_(penalty_cost)
      {}
  };
  std::vector<SpacePenalty>left_space_penalty_factor_;

  void set_left_space_penalty_factor(const char *factor_str);
  int get_space_penalty_cost(const Node *rNode) const;

 public:

  bool open(const Param &param);
  void close();
  void clear() {}

  const char *what() { return what_.str(); }

  size_t left_size()  const { return static_cast<size_t>(lsize_); }
  size_t right_size() const { return static_cast<size_t>(rsize_); }

  void set_left_size(size_t lsize)  { lsize_ = lsize; }
  void set_right_size(size_t rsize) { rsize_ = rsize; }

  inline int transition_cost(unsigned short rcAttr,
                             unsigned short lcAttr) const {
    return matrix_[rcAttr + lsize_ * lcAttr];
  }

  int cost(const Node *lNode, const Node *rNode) const;

  // access to raw matrix
  short *mutable_matrix() { return &matrix_[0]; }
  const short *matrix() const { return &matrix_[0]; }

  bool openText(const char *filename);
  bool open(const char *filename, const char *white_space_penalty_info = "", const char *mode = "r");

  bool is_valid(size_t lid, size_t rid) const {
    return (lid >= 0 && lid < rsize_ && rid >= 0 && rid < lsize_);
  }

  static bool compile(const char *, const char *);

  explicit Connector():
      cmmap_(new Mmap<short>), matrix_(0), lsize_(0), rsize_(0) {}

  virtual ~Connector() { this->close(); }
};
}
#endif  // MECAB_CONNECTOR_H_
