import os
from collections import defaultdict
from utils_ocr.utils import StringDistance, extract_digit


class AddressCorrection:
    """
    Address correction with phrase compare
    """
    def __init__(self, cost_dict_path=None, provinces_path=None, districts_path=None, wards_path=None, underwards_path=None):
        # dir_path = os.path.dirname(os.path.realpath(__file__))
        if cost_dict_path is None:
            cost_dict_path = os.path.join('utils_ocr', 'data_address_provinceVN', 'cost_char_dict.txt')
        if provinces_path is None:
            provinces_path = os.path.join('utils_ocr', 'data_address_provinceVN', 'provinces.txt')
        if districts_path is None:
            districts_path = os.path.join('utils_ocr', 'data_address_provinceVN', 'districts.txt')
        if wards_path is None:
            wards_path = os.path.join('utils_ocr', 'data_address_provinceVN', 'wards2.txt')
        if underwards_path is None:
            underwards_path = os.path.join('utils_ocr', 'data_address_provinceVN', 'underwards.txt')
        self.string_distance = StringDistance(cost_dict_path=cost_dict_path)
        self.provinces = []
        self.districts = defaultdict(list)
        self.wards = defaultdict(list)
        self.underwards = defaultdict(list)
        self.province_wards = defaultdict(list)
        with open(provinces_path, 'r', encoding='UTF-8') as f:
            for line in f:
                entity = line.strip()
                if not entity:
                    break
                entity = entity.split('|')
                self.provinces.extend(entity)

        with open(districts_path, 'r', encoding='UTF-8') as f:
            for line in f:
                entity = line.strip()
                districts, provinces = entity.split('\t')
                districts = districts.split('|')
                provinces = provinces.split('|')
                for province in provinces:
                    self.districts[province].extend(districts)
        with open(wards_path, 'r', encoding='UTF-8') as f:
            for line in f:
                entity = line.strip()
                wards, districts, provinces = entity.split('\t')
                districts = districts.split('|')
                wards = wards.split('|')
                provinces = provinces.split('|')
                for province in provinces:
                    for district in districts:
                        self.wards[(province, district)].extend(wards)
                        # correct for lack of district
                        self.province_wards[province].extend([ward for ward in wards if len(ward) > 3])
        with open(underwards_path, 'r', encoding='UTF-8') as f:
            for line in f:
                entity = line.strip()
                underward, ward, district, province = entity.split('\t')
                self.underwards[(province, district, ward)].append(underward)
        self.provinces = tuple(set(self.provinces))
        self.districts = {k:tuple(set(self.districts[k])) for k in self.districts}
        self.wards = {k:tuple(set(self.wards[k])) for k in self.wards}
        self.underwards = {k:tuple(set(self.underwards[k])) for k in self.underwards}
        self.province_wards = {k:tuple(set(self.province_wards[k])) for k in self.province_wards}

    def correct(self, phrase, correct_phrases, nb_candidates=2, distance_threshold=40):
        candidates = [(None, distance_threshold)] * nb_candidates
        max_diff_length = distance_threshold
        for correct_phrase in correct_phrases:
            if abs(len(phrase) - len(correct_phrase)) >= max_diff_length:
                continue
            if extract_digit(correct_phrase) != extract_digit(phrase):
                distance = 100
            else:
                distance = self.string_distance.distance(phrase, correct_phrase)
            if distance < candidates[-1][1]:
                candidates[-1] = (correct_phrase, distance)
                candidates.sort(key=lambda x:x[1])
        return candidates

    def _wards_correction(self, tokens, prefix_province, province, prefix_district, district,
                          current_district_index, current_distance, current_result_distance):
        result = None
        result_distance = current_result_distance
        district_normalized = district + ',' if len(prefix_district) == 0 else \
            '{} {},'.format(prefix_district, district)
        province_normalized = province if len(prefix_province) == 0 else \
            '{} {}'.format(prefix_province, province)
        for wards_index in range(max(0, current_district_index - 4), current_district_index):
            phrase = ' '.join(tokens[wards_index:current_district_index])
            correct_wards = self.wards.get((province, district), tuple())
            if len(phrase) < 8:
                distance_th = 15
            else:
                distance_th = 20
            wards_candidates = self.correct(phrase, correct_wards, distance_threshold=distance_th, nb_candidates=2)
            for wards, wards_distance in wards_candidates:
                if wards and len(wards) < 5:
                    if district == 'tp':
                        wards_distance *= 3
                    else:
                        wards_distance *= 2
                new_distance = current_distance + wards_distance
                if new_distance > result_distance or wards is None:
                    continue

                def check_prefix():
                    new_wards_index = None
                    prefix_wards = None
                    distance = new_distance
                    if wards_index < 1:
                        return new_wards_index, prefix_wards, distance
                    if tokens[wards_index - 1] == 'p':
                        prefix_wards = 'p'
                        new_wards_index = wards_index - 1
                        return new_wards_index, prefix_wards, distance
                    if tokens[wards_index - 1] == 'x':
                        prefix_wards = 'x'
                        new_wards_index = wards_index - 1
                        return new_wards_index, prefix_wards, distance
                    if tokens[wards_index - 1] == 'xã':
                        prefix_wards = 'xã'
                        new_wards_index = wards_index - 1
                        return new_wards_index, prefix_wards, distance
                    if tokens[wards_index - 1] == 'tt':
                        prefix_wards = 'tt'
                        new_wards_index = wards_index - 1
                        return new_wards_index, prefix_wards, distance
                    d = self.string_distance.distance(tokens[wards_index - 1], 'phường')
                    if d < 10:
                        prefix_wards = 'phường'
                        new_wards_index = wards_index - 1
                        distance = d + new_distance
                        return new_wards_index, prefix_wards, distance
                    # d = self.string_distance.distance(tokens[wards_index - 1], 'thị trấn')
                    # if d <= 20:
                    #     prefix_wards = 'thị trấn'
                    #     new_wards_index = wards_index - 1
                    #     distance = d + new_distance
                    #     return new_wards_index, prefix_wards, distance
                    # if wards_index < 2:
                    #     return new_wards_index, prefix_wards, distance
                    # d = self.string_distance.distance(' '.join(tokens[wards_index - 2:wards_index]), 'thị trấn')
                    # if d <= 20:
                    #     prefix_wards = 'thị trấn'
                    #     new_wards_index = wards_index - 2
                    #     distance = d + new_distance
                    #     return new_wards_index, prefix_wards, distance
                    return new_wards_index, prefix_wards, distance
                new_wards_index, prefix_wards, _ = check_prefix()
                if new_wards_index is None:
                    new_wards_index = wards_index
                wards_normalized = wards + ',' if prefix_wards is None else '{} {},'.format(prefix_wards, wards)
                address_composition = [wards_normalized, district_normalized, province_normalized]
                if new_wards_index > 0:
                    underwards_tokens = tokens[:new_wards_index]
                    correct_underwards = self.underwards.get((province, district, wards), tuple())
                    corrected_underwards = None
                    under_wards_index = None
                    for i in range(len(underwards_tokens)-1, max(-1, len(underwards_tokens)-5), -1):
                        if not tokens[i].isalpha():
                            break
                        underwards_phrase = ' '.join(underwards_tokens[i:wards_index])
                        th_distance = 15 if len(underwards_phrase) < 6 else 20
                        candidates = self.correct(underwards_phrase, correct_underwards,
                            nb_candidates=1, distance_threshold=th_distance)
                        if candidates[0][0] is not None:
                            corrected_underwards = candidates[0][0]
                            under_wards_index = i
                            break
                    if corrected_underwards is not None:
                        prefix_address = ' '.join(tokens[:under_wards_index] + [corrected_underwards + ','])
                    else:
                        prefix_address = ' '.join(tokens[:new_wards_index]) + ','
                    address_composition = [prefix_address] + address_composition
                result = ' '.join(address_composition)
                result_distance = new_distance
        if result is None and current_distance < 20:
            if not prefix_district:
                result_distance = 45
            else:
                result_distance = 40
            prefix_address = ' '.join(tokens[:current_district_index]) + ','
            result = ' '.join([prefix_address, district_normalized , province_normalized])
        return result, result_distance

    def _district_correction(self, tokens, prefix_province, province,
                             current_province_index, current_distance, current_result_distance):
        result = None
        normalized_province = '{} {}'.format(prefix_province, province) if prefix_province else province
        result_distance = current_result_distance
        early_stop_threshold = 0
        stop_correction = False
        for district_index in range(max(0, current_province_index - 4), current_province_index):
            phrase = ' '.join(tokens[district_index:current_province_index])
            # correct that lack of district
#            if province in ['hn', 'hà nội', 'hcm', 'hồ chí minh']:
#                correct_wards = self.province_wards[province]
#                ward_candidates = self.correct(phrase, correct_districts, nb_candidates=3, distance_threshold=10)
#                for ward, distance_ward in ward_candidates:
#                    def check_prefix():
#                        new_district_index = None
#                        prefix_district = None
#                        distance = new_distance
#                        if district_index <= 0:
#                            return new_district_index, prefix_district, distance
#                        d = self.string_distance.distance(tokens[district_index - 1], 'huyện')
#                        if d <= 2:
#                            prefix_district = 'huyện'
#                            new_district_index = district_index - 1
#                            distance = d + new_distance
#                            return new_district_index, prefix_district, distance
#                        if tokens[district_index - 1] == 'q':
#                            prefix_district = 'q'
#                            new_district_index = district_index - 1
#                            return new_district_index, prefix_district, distance
#                        if tokens[district_index - 1] == 'quận':
#                            prefix_district = 'quận'
#                            new_district_index = district_index - 1
#                            return new_district_index, prefix_district, distance
#                        if tokens[district_index - 1] == 'tp':
#                            prefix_district = 'tp'
#                            new_district_index = district_index - 1
#                            return new_district_index, prefix_district, distance
#                        if tokens[district_index - 1] == 'tt':
#                            prefix_district = 'tt'
#                            new_district_index = district_index - 1
#                            return new_district_index, prefix_district, distance
#                        if tokens[district_index - 1] == 'tx':
#                            prefix_district = 'tx'
#                            new_district_index = district_index - 1
#                            return new_district_index, prefix_district, distance
#
#                    
#            # correct full ward, district, province
            correct_districts = self.districts.get(province, tuple())
            district_candidates = self.correct(phrase, correct_districts, nb_candidates=3)
            for district, distance_district in district_candidates:
                if district and (len(phrase) < 5 or len(district) < 5):
                    distance_district *= 3
                new_distance = current_distance + distance_district
                if new_distance >= result_distance or district is None:
                    continue
                if district_index > 0:
                    result_candidate, result_distance_candidate = self._wards_correction(
                        tokens, prefix_province, province, '', district, district_index,
                        new_distance, current_result_distance
                    )
                    if result_distance > result_distance_candidate:
                        result = result_candidate
                        result_distance = result_distance_candidate
                    def check_prefix():
                        new_district_index = None
                        prefix_district = None
                        distance = new_distance
                        if district_index <= 0:
                            return new_district_index, prefix_district, distance
                        d = self.string_distance.distance(tokens[district_index - 1], 'huyện')
                        if d <= 2:
                            prefix_district = 'huyện'
                            new_district_index = district_index - 1
                            distance = d + new_distance
                            return new_district_index, prefix_district, distance
                        if tokens[district_index - 1] == 'q':
                            prefix_district = 'q'
                            new_district_index = district_index - 1
                            return new_district_index, prefix_district, distance
                        if tokens[district_index - 1] == 'quận':
                            prefix_district = 'quận'
                            new_district_index = district_index - 1
                            return new_district_index, prefix_district, distance
                        if tokens[district_index - 1] == 'tp':
                            prefix_district = 'tp'
                            new_district_index = district_index - 1
                            return new_district_index, prefix_district, distance
                        if tokens[district_index - 1] == 'tt':
                            prefix_district = 'tt'
                            new_district_index = district_index - 1
                            return new_district_index, prefix_district, distance
                        if tokens[district_index - 1] == 'tx':
                            prefix_district = 'tx'
                            new_district_index = district_index - 1
                            return new_district_index, prefix_district, distance
                        d = self.string_distance.distance(tokens[district_index - 1], 'thành phố')
                        if d < 30:
                            prefix_district = 'thành phố'
                            new_district_index = district_index - 1
                            distance = d + new_distance
                            return new_district_index, prefix_district, distance
                        if district_index < 2:
                            return new_district_index, prefix_district, distance
                        d = self.string_distance.distance(' '.join(tokens[district_index - 2: district_index]), 'thành phố')
                        if d <= 20:
                            prefix_district = 'thành phố'
                            new_district_index = district_index - 2
                            distance = d + new_distance
                            return new_district_index, prefix_district, distance
                        d = self.string_distance.distance(' '.join(tokens[district_index - 2: district_index]), 'thị xã')
                        if d <= 20:
                            prefix_district = 'thị xã'
                            new_district_index = district_index - 2
                            distance = d + new_distance
                            return new_district_index, prefix_district, distance
                        return new_district_index, prefix_district, distance
                    new_district_index, prefix_district, new_distance = check_prefix()
                    if new_district_index is None:
                        continue
                    if new_district_index > 0:
                        result_candidate, result_distance_candidate = self._wards_correction(
                            tokens, prefix_province, province, prefix_district, district,
                            new_district_index, new_distance, current_result_distance
                        )
                        if result_distance > result_distance_candidate:
                            result = result_candidate
                            result_distance = result_distance_candidate
                    else:
                        if new_distance < result_distance:
                            result_distance = new_distance
                            normalized_district = '{} {}'.format(prefix_district, district)
                            result = '{}, {}'.format(normalized_district, normalized_province)
                elif new_distance < result_distance:
                    result = district + ', ' + normalized_province
                    result_distance = new_distance
                if distance_district <= early_stop_threshold:
                    stop_correction = True
                    break
            if stop_correction:
                break
        return result, result_distance

    def _province_correction(self, tokens):
        result_distance = 1000
        result = None
        nb_of_tokens = len(tokens)
        early_stop_threshold = 0
        stop_correction = False
        for index_province in range(max(0, nb_of_tokens - 4), nb_of_tokens):
            phrase = ' '.join(tokens[index_province:])
            province_candidates = self.correct(phrase, self.provinces)
            for province, distance_province in province_candidates:
                if distance_province > result_distance or province is None:
                    continue
                result_candidate, result_distance_candidate = self._district_correction(
                    tokens, '', province, index_province,
                    distance_province, result_distance
                )
                if result_distance_candidate < result_distance:
                    result_distance = result_distance_candidate
                    result = result_candidate
                if index_province > 0:
                    if tokens[index_province-1] in ['tp', 't/p']:
                        if index_province <= 1:
                            result = 'tp ' + province
                            result_distance = distance_province
                            continue
                        result_candidate, result_distance_candidate = self._district_correction(
                            tokens, 'tp', province, index_province - 1,
                            distance_province, result_distance
                        )
                        if result_distance_candidate < result_distance:
                            result_distance = result_distance_candidate
                            result = result_candidate
                    elif tokens[index_province].startswith('tp'):
                        if index_province <= 1:
                            result = 'tp ' + province
                            result_distance = distance_province
                            continue
                        result_candidate, result_distance_candidate = self._district_correction(
                            tokens, 'tp', province, index_province,
                            distance_province, result_distance
                        )
                        if result_distance_candidate < result_distance:
                            result_distance = result_distance_candidate
                    elif self.string_distance.distance(tokens[index_province-1], 'tỉnh') < 10:
                        if index_province <= 1:
                            result = 'tỉnh ' + province
                            result_distance = distance_province
                            continue
                        result_candidate, result_distance_candidate = self._district_correction(
                            tokens, 'tỉnh', province, index_province-1,
                            distance_province, result_distance
                        )
                        if result_distance_candidate < result_distance:
                            result_distance = result_distance_candidate
                            result = result_candidate
                    elif index_province > 1 and self.string_distance.distance(' '.join(tokens[index_province-2:index_province]), 'thành phố') < 20:
                        if index_province <= 1:
                            result = 'thành phố ' + province
                            result_distance = distance_province
                            continue
                        result_candidate, result_distance_candidate = self._district_correction(
                            tokens, 'thành phố', province, index_province-2,
                            distance_province, result_distance
                        )
                        if result_distance_candidate < result_distance:
                            result_distance = result_distance_candidate
                            result = result_candidate
                if index_province <= 0:
                    if distance_province < result_distance:
                        result_distance = distance_province
                        result = province
                if distance_province <= early_stop_threshold:
                    stop_correction = True
                    break
            if stop_correction:
                break
        return result, result_distance

    def address_correction(self, address, correct_th=50):
        """
        Address should be in format: Ngõ ngách... đường... quận/huyện...tỉnh/thành phố
        and only contain characters
        Return: (corrected_address: str, distance: integer)
            corrected_address: address after corrected. In case address can't corrected, return
            input address
            distance: distance between corrected address and input address. In case address
            can't correct, return -1
        """
        if not isinstance(address, str):
            raise ValueError('Address must be a string')
        address = address.replace('.', ' ').replace('-', ' ').replace("  ", '')
        tokens = address.split()
        result, distance_result = self._province_correction(tokens)
        if distance_result <= correct_th:
            # nb_of_comma = result.count(',')
            # if nb_of_comma > 2:
            #     prefix_number = ('tdp', 'ấp', 'phố', 'số', 'đội', 'xóm', 'khu', 'ngách', 'đường', 'tổ', 'ngõ', 'phường', 'khóm', 'thôn') # thiếu string "số"
            #     tokens = result.split()
            #     for i in range(1, min(5, len(tokens) - 1)):
            #         have_comma = ',' in tokens[i]
            #         if (not have_comma and not tokens[i].isalpha()) or (have_comma and not tokens[i][:-1].isalpha()):
            #             corrected_token = self.correct(tokens[i-1], prefix_number, nb_candidates=1, distance_threshold=20)[0]
            #             if corrected_token[0] is not None:
            #                 tokens[i-1] = corrected_token[0]
            #         if have_comma:
            #             break
            #     result = ' '.join(tokens)
            #     result = result.replace('.', ' ').replace(',,', ',')
            #     result = result.replace('.', ' ').replace('tp,', 'tp')
            #     return result
            # else:
            result = result.replace('.', ' ').replace(',,', ',')
            result = result.replace('.', ' ').replace('tp,', 'tp')
            return result
        else:
            return address

    def address_extraction(self, address):
        components = [" ", " ", " ", " "]
        list_components = [self.districts, self.wards]
        current_list = self.provinces
        tokens = address.split(',')
        k = len(tokens)
        for i in range(3):
            for j in reversed(range(k)):
                token_norm = tokens[j].replace("  ", '')
                token = self.remove_prefixed(token_norm, ['tp', 'tỉnh', 'thành phố', 'thị xã', 'tx', 'huyện',
                                                         'quận', 'thị trấn', 'tt', 'xã', 'q ', 'x '])
                token_fixes = [token_norm.strip(), token, 'q.' + token,
                               'x.' + token, 'p.' + token, token.replace(' ', '.')]
                token_verify = False
                for token_fix in token_fixes:
                    if token_fix in current_list:
                        token_verify = True
                        components[i] = token_fix
                        if i == 0:
                            current_list = list_components[i].get(components[0], tuple())
                        elif i == 1:
                            current_list = list_components[i].get((components[0], components[1]), tuple())
                        k = j
                if token_verify:
                    break
        components[3] = " ".join(tokens[:k])
        for i, token in enumerate(components):
            components[i] = token.title()
        return components

    def remove_prefixed(self, phrase, prefixeds):
        phrase = phrase.strip()
        for prefixed in prefixeds:
            phrase = phrase.replace(prefixed, " ")
        if not phrase[-1].isnumeric():
            phrase = phrase.replace("phường", " ")
            phrase = phrase.replace("p ", " ")
        phrase = phrase.strip()
        return phrase

    def special_fix_for_hue(self, address):
        province = ''
        district = ''
        ward = ''
        hue_phrases = ['huê', 'huế', 'húê', 'hue', 'húe', 'hué',
                      'thưa', 'thừa', 'thua', 'thùa', 'thưà', 'thuà'
                      'thiên', 'thien']
        is_hue_address = False
        for hue_phrase in hue_phrases:
            if hue_phrase in address:
                is_hue_address = True
        if not is_hue_address:
            return address

        province = 'thừa thiên huế'
        address = address.replace('.', ' ').replace('-', ' ').replace("  ", '')
        tokens = address.split()
