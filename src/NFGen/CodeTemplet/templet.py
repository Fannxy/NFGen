"""Three supported code templet.
"""

templet_spdz = '''
@types.vectorize
def general_non_linear_func(x):
    """Version2 of general non linear function.

    Args:
        x (Sfixed): the input secret value.
        coeffA (plain-text 2d python list): The plain-text coefficient of specific non-linear functions.
        breaks (plain-text 1d python list): The plain-rext break points of specific functions.

    Returns:
        Sfixed: f(x) value of specific non-lnear function f.
    """
    # insert here
    
    m = len(coeffA)
    k = len(coeffA[0])
    degree = k-1
    
    pre_muls = floatingpoint.PreOpL(lambda a,b,_: a * b, [x] * degree)

    poss_res = [0]*m
    for i in range(m):
        poss_res[i] = coeffA[i][0] * scaler[i][0]
        for j in range(degree):
            poss_res[i] += coeffA[i][j+1] * pre_muls[j] * scaler[i][j+1]

    comp = sfix.Array(m)
    for i in range(m):
        comp[i] = (x >= breaks[i])
        
    cipher_index = Array(n, sfix)
    @for_range_opt(n-1)
    def _(i):
        cipher_index[i] = comp[i+regint(1)]
        cipher_index[i] = comp[i]*(comp[i] - cipher_index[i])

    return sfix.dot_product(cipher_index, poss_res)
'''

templet_flp_spdz = '''
@types.vectorize
def general_non_linear_flp(x, flen=48, klen=None):
    """care the input x is floating point.
    """
    # insert here
    
    if klen is None:
        klen = 2*flen

    m = len(coeffA)
    k = len(coeffA[0])
    degree = k-1
    
    pre_muls = floatingpoint.PreOpL(lambda a,b,_: a * b, [x] * degree)

    poss_res = [0]*m
    for i in range(m):
        poss_res[i] = FLP_to_FXP(coeffA[i][0], flen, klen)
        for j in range(degree):
            poss_res[i] += FLP_to_FXP(coeffA[i][j+1] * pre_muls[j], flen, klen)

    sfix_x = FLP_to_FXP(x, flen, klen)
    comp = sfix.Array(m)
    for i in range(m):
        comp[i] = (sfix_x >= breaks[i])
    cipher_index = bb.get_last_one(comp) 
    return sfix.dot_product(cipher_index, poss_res)
'''

templet_privpy_py = '''
@pp.local_import("numpy", "np")
def general_non_linear_func(x):
    import pnumpy as pnp
    
    def _calculate_kx(x, k):
        items = pnp.transpose(pnp.tile(x, (k, 1)))
        items[:, 0] = pp.sfixed(1)

        shift = 1
        while shift < k:
            items[:, shift:] *= items[:, :len(items[0])-shift]
            shift *= 2
        return items

    def _fetch_index(x, breaks):
        if isinstance(x, pp.FixedArr):
            x = pnp.transpose(pnp.tile(x, (len(breaks), 1)))
        breaks = np.tile(breaks, (len(x), 1))

        cipher_comp = x >= breaks
        cipher_index = pnp.util.get_last_one(cipher_comp, method="normal")

        return cipher_index
    
    # insert here

    breaks = np.array(breaks)
    coeffA = np.array(coeffA)
    scalerA = np.array(scaler)

    k = int(len(coeffA[0]))
    cipher_index = _fetch_index(x, breaks)
    coeff = pnp.dot(cipher_index, coeffA)
    scaler = pnp.dot(cipher_index, scalerA)
    x_items = _calculate_kx(x, k)
    tmp_res = x_items * coeff
    res = pnp.sum(tmp_res * scaler, axis=1)
    return res
'''

templet_privpy_cpp = '''
bool SS4Runner::general_non_linear_func(const size_type length,
                      TypeSet::FNumberArr *result,
                      const TypeSet::FNumberArr *num,
                      bool use_current_thread) {
  check_runner_terminate_status(false);
  TypeSet::FNUMT *num_x = num->get_x(), *num_x_ = num->get_x_();
  TypeSet::FNUMT *result_x = result->get_x(), *result_x_ = result->get_x_();
  
  # insert here
  
  // expand num and breaks to outter_comparision lenghth.
  const size_type expand_length = length * M;

  TypeSet::BitArr bit_arr(expand_length);
  TypeSet::FNumberArr sub_arr(expand_length);
  TypeSet::FNumberArr breaks(expand_length);
  TypeSet::FNumberArr findex_arr(expand_length);
  TypeSet::FNumberArr ctmp(expand_length);

  double *ftmp = new double[expand_length];

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < M; j++) {
      ftmp[i * M + j] = Breaks[j];
      sub_arr.x[i * M + j] = num_x[i];
      sub_arr.x_[i * M + j] = num_x_[i];
    }
  }

  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &breaks);
  sub<TypeSet::FNUMT>(expand_length, &sub_arr, &breaks, &sub_arr);
  extractv<TypeSet::FNUMT>(sub_arr, -1, expand_length, &bit_arr);
  // The index_arr now stores the pair-wise comparision result.
  // for(int i=0; i<expand_length; i++) ftmp[i] = 1.0;
  std::fill_n(ftmp, expand_length, 1.0);
  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &ctmp);
  ot<TypeSet::FNUMT>(expand_length, &findex_arr, &ctmp, &bit_arr);
  // Quick version get_last_one.
  // for (int i = 0; i < expand_length; i++) ftmp[i] = 0.0;
  std::fill_n(ftmp, expand_length, 0.0);
  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &ctmp);

  // Get last one for cipher-index.
  for (size_type i = 0; i < length; i++) {
    memcpy(ctmp.x + i * M, findex_arr.x + i * M + 1, sizeof(TypeSet::FNUMT) * (M - 1));
    memcpy(ctmp.x_ + i * M, findex_arr.x_ + i * M + 1, sizeof(TypeSet::FNUMT) * (M - 1));
  }
  sub<TypeSet::FNUMT>(expand_length, &findex_arr, &findex_arr, &ctmp);

  const size_type config_length = length * K;
  TypeSet::FNumberArr coeff(config_length);
  TypeSet::FNumberArr scaler(config_length);
  TypeSet::FNumberArr Ocoeff(M * K);
  TypeSet::FNumberArr Oscaler(M * K);
  map2numv<double, TypeSet::FNUMT>(CoeffA, M * K, &Ocoeff);
  map2numv<double, TypeSet::FNUMT>(Scaler, M * K, &Oscaler);
  size_type shape1[2] = {length, M};
  size_type shape2[2] = {M, K};
  inner_product<TypeSet::FNUMT>(&coeff, shape1, shape2, &findex_arr, &Ocoeff);
  inner_product<TypeSet::FNUMT>(&scaler, shape1, shape2, &findex_arr, &Oscaler);

  // Calculate [1, x, x^2, ..., x^k] with log(k) round multiplications.

  // 1. Value initialization, get [1, x, x, ..., x].
  TypeSet::FNumberArr x_items(config_length);
  // TypeSet::FNUMT *x_items_x = x_items->get_x(), *x_items_x_ = x_items->get_x_();
  size_type *begin_list = new size_type[length];

  for (size_type i = 0; i < length; i++) {
    if (_ROLE == ROLE_S1) {
      x_items.x[i * K] = privpy::SS4::scale2big<TypeSet::FNUMT, double>(1.0);
      x_items.x_[i * K] = 0;
    } else if (_ROLE == ROLE_S2) {
      x_items.x[i * K] = 0;
      x_items.x_[i * K] = privpy::SS4::scale2big<TypeSet::FNUMT, double>(1.0);
    } else if (_ROLE == ROLE_SA) {
      x_items.x[i * K] = x_items.x_[i * K] = 0;
    } else if (_ROLE == ROLE_SB) {
      x_items.x[i * K] = privpy::SS4::scale2big<TypeSet::FNUMT, double>(1.0);
      x_items.x_[i * K] = x_items.x[i * K];
    }
    begin_list[i] = i * K;
  }
  for (size_type i = 0; i < length; i++) {
    for (size_type j = 1; j < K; j++) {
      x_items.x[i * K + j] = num_x[i];
      x_items.x_[i * K + j] = num_x_[i];
    }
  }

  // 2. Construct the corresponding x_items [1, x, x^2, ..., x^k].
  size_type shift = 1;
  while (shift < K) {
    size_type *start_list = new size_type[length];
    for (int i = 0; i < length; i++) start_list[i] = i * K + shift;
    size_type tmp_length = K - shift;

    TypeSet::FNumberArr arr1(tmp_length * length);
    TypeSet::FNumberArr arr2(tmp_length * length);

    for (size_type i = 0; i < length; i++) {
      memcpy(arr1.x + i * tmp_length, x_items.x + start_list[i], sizeof(TypeSet::FNUMT) * tmp_length);
      memcpy(arr1.x_ + i * tmp_length, x_items.x_ + start_list[i], sizeof(TypeSet::FNUMT) * tmp_length);
      memcpy(arr2.x + i * tmp_length, x_items.x + begin_list[i], sizeof(TypeSet::FNUMT) * tmp_length);
      memcpy(arr2.x_ + i * tmp_length, x_items.x_ + begin_list[i], sizeof(TypeSet::FNUMT) * tmp_length);
    }
    // Cipher-text multiplication. arr1 *= arr2;
    if (!mul<TypeSet::FNUMT>(tmp_length * length, &arr1, &arr1, &arr2)) {
      return false;
    }
    // Update x_items.
    for (size_type i = 0; i < length; i++) {
      memcpy(x_items.x + start_list[i], arr1.x + (i * tmp_length), sizeof(TypeSet::FNUMT) * tmp_length);
      memcpy(x_items.x_ + start_list[i], arr1.x_ + (i * tmp_length), sizeof(TypeSet::FNUMT) * tmp_length);
    }
    shift *= 2;
  }

  // Calculate the final result, sum(x*coeff*scaler, axis=1); The order is important, first coeff and then the scaler.

  // tri_mul<TypeSet::FNUMT>(config_length, &x_items, &x_items, &coeff, &scaler);
  mul<TypeSet::FNUMT>(config_length, &x_items, &x_items, &coeff);
  mul<TypeSet::FNUMT>(config_length, &x_items, &x_items, &scaler);

  double *ftmp2 = new double[length];
  for (int i = 0; i < length; i++) ftmp2[i] = 0.0;

  map2numv<double, TypeSet::FNUMT>(ftmp2, length, result);
  for (size_type i = 0; i < length; i++) {
    for (size_type j = 0; j < K; j++) {
      result_x[i] += x_items.x[i * K + j];
      result_x_[i] += x_items.x_[i * K + j];
    }
  }
  delete[] ftmp2;

  return true;
}
'''

templet_privpy_cpp2 = '''
bool SS4Runner::general_non_linear_func_calculateCoeff(const size_type length,
                                     TypeSet::FNumberArr *coeff,
                                     TypeSet::FNumberArr *scaler,
                                     const TypeSet::FNumberArr *num) {
  check_runner_terminate_status(false);
  TypeSet::FNUMT *num_x = num->get_x(), *num_x_ = num->get_x_();
  # insert here

  // PART1: comparision and find the corresponding coeff and scaler.
  // expand num and breaks to outter_comparision lenghth.
  const size_type expand_length = length * M;
  TypeSet::BitArr bit_arr(expand_length);
  TypeSet::FNumberArr sub_arr(expand_length);
  TypeSet::FNumberArr breaks(expand_length);
  TypeSet::FNumberArr findex_arr(expand_length);
  TypeSet::FNumberArr ctmp(expand_length);

  double *ftmp = new double[expand_length];

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < M; j++) {
      ftmp[i * M + j] = Breaks[j];
      sub_arr.x[i * M + j] = num_x[i];
      sub_arr.x_[i * M + j] = num_x_[i];
    }
  }
  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &breaks);
  sub<TypeSet::FNUMT>(expand_length, &sub_arr, &breaks, &sub_arr);
  extractv<TypeSet::FNUMT>(sub_arr, -1, expand_length, &bit_arr);
  // The index_arr now stores the pair-wise comparision result.
  // for(int i=0; i<expand_length; i++) ftmp[i] = 1.0;
  std::fill_n(ftmp, expand_length, 1.0);
  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &ctmp);
  ot<TypeSet::FNUMT>(expand_length, &findex_arr, &ctmp, &bit_arr);
  // Quick version get_last_one.
  // for (int i = 0; i < expand_length; i++) ftmp[i] = 0.0;
  std::fill_n(ftmp, expand_length, 0.0);
  map2numv<double, TypeSet::FNUMT>(ftmp, expand_length, &ctmp);

  // Get last one for cipher-index.
  for (size_type i = 0; i < length; i++) {
    memcpy(ctmp.x + i * M, findex_arr.x + i * M + 1, sizeof(TypeSet::FNUMT) * (M - 1));
    memcpy(ctmp.x_ + i * M, findex_arr.x_ + i * M + 1, sizeof(TypeSet::FNUMT) * (M - 1));
  }
  sub<TypeSet::FNUMT>(expand_length, &findex_arr, &findex_arr, &ctmp);

  TypeSet::FNumberArr Ocoeff(M * K);
  TypeSet::FNumberArr Oscaler(M * K);
  map2numv<double, TypeSet::FNUMT>(CoeffA, M * K, &Ocoeff);
  map2numv<double, TypeSet::FNUMT>(Scaler, M * K, &Oscaler);
  size_type shape1[2] = {length, M};
  size_type shape2[2] = {M, K};
  inner_product<TypeSet::FNUMT>(coeff, shape1, shape2, &findex_arr, &Ocoeff);
  inner_product<TypeSet::FNUMT>(scaler, shape1, shape2, &findex_arr, &Oscaler);

  return true;
}

bool SS4Runner::general_non_linear_func(const size_type length,
                      TypeSet::FNumberArr *result,
                      const TypeSet::FNumberArr *num,
                      bool use_current_thread) {
  check_runner_terminate_status(false);
  TypeSet::FNUMT *result_x = result->get_x(), *result_x_ = result->get_x_();

  # insert here
  // PART1: comparision and find the corresponding coeff and scaler.
  // expand num and breaks to outter_comparision lenghth.
  const size_type config_length = length * K;
  TypeSet::FNumberArr coeff(config_length);
  TypeSet::FNumberArr scaler(config_length);
  TypeSet::FNumberArr x_items(config_length);

  auto futurecoeff = threadPool.submit([&] { this->general_non_linear_func_calculateCoeff(length, &coeff, &scaler, num); });
  auto futurekx = threadPool.submit([&] { this->calculateKx(length, &x_items, num, K); });
  threadPool.wait_for_tasks();

  mul<TypeSet::FNUMT>(config_length, &x_items, &x_items, &coeff);
  mul<TypeSet::FNUMT>(config_length, &x_items, &x_items, &scaler);

  double *ftmp2 = new double[length];
  for (int i = 0; i < length; i++) ftmp2[i] = 0.0;

  map2numv<double, TypeSet::FNUMT>(ftmp2, length, result);
  for (size_type i = 0; i < length; i++) {
    for (size_type j = 0; j < K; j++) {
      result_x[i] += x_items.x[i * K + j];
      result_x_[i] += x_items.x_[i * K + j];
    }
  }
  delete[] ftmp2;

  return true;
}
'''


templet_plain = '''
def general_non_linear_func(x):
    """Plaintext evaluation
    """
    def calculate_kx(x, k):
        items = np.transpose(np.tile(x, (k, 1)))
        items[:, 0] = 1
        
        # example: shift=1
        # items[i]       : 1  x  x  x  x  x
        # items[i-shift] : s  1  x  x  x  x
        # next_items     : 1  x  x2 x2 x2 x2
        shift = 1
        while shift < k:
            items[:, shift:] *= items[:, :-shift] # modified
            items[np.isnan(items)] = 0
            shift *= 2
            
        return items
    # insert here
    
    breaks = np.array(breaks)
    coeffA = np.array(coeffA)
    scalerA = np.array(scaler)    

    if(len(x.shape) < 2):
        x = x[:, np.newaxis]
    x_index = x >= breaks
    x_index= np.sum(x_index, axis=1) - 1
    coeff = coeffA[x_index]
    scale = scalerA[x_index]
    x = np.squeeze(x)
    x_items = np.round((calculate_kx(x, len(coeff[0])) * coeff), 14)
    tmp = np.round(np.sum(x_items * scale, axis=1), 14)
    
    return tmp
'''