exp_interface_1="""
<p><strong>The Mystery Author can be described as having the following styles:</strong></p>
<p><strong>For each style, we show how much it applies for each of the candidate authors</strong></p>
<p><strong>Style 1:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-1]</em></span></p>
<p><strong>Style 2:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-2]</em></span></p>
<p><strong>Style 3:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-3]</em></span></p>
<p><br></p>
<p><strong>Here is how much these styles resemble each of the candidate authors:</strong>(darker color == high resemblance and light color == low resemblance)</p>
<p><strong>Note: The darker the row color the more likly the corresponding candidate to be similar to the Query author.</strong></p>
<p><br></p>
<table style="width: 70%;">
    <tbody>
        <tr>
            <td style="width: 30%; background-color: rgb(255, 255, 255);"><br></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Style 1</strong></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Style 2</strong></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Style 3</strong></td>
        </tr>
        <tr>
            <td style="width: 30%; background-color: rgb(209, 213, 216);">Candidate Author 1</td>
            <td style="width: 14%; background-color: [cand-1-style-1];"><br></td>
            <td style="width: 14%; background-color: [cand-1-style-2];"><br></td>
            <td style="width: 14%; background-color: [cand-1-style-3];"><br></td>
        </tr>
        <tr>
            <td style="width: 30%; background-color: rgb(209, 213, 216);">Candidate Author 2</td>
            <td style="width: 14%; background-color: [cand-2-style-1];"><br></td>
            <td style="width: 14%; background-color: [cand-2-style-2];"><br></td>
            <td style="width: 14%; background-color: [cand-2-style-3];"><br></td>
        </tr>
        <tr>
            <td style="width: 30%; background-color: rgb(209, 213, 216);">Candidate Author 3</td>
            <td style="width: 14%; background-color: [cand-3-style-1];"><br></td>
            <td style="width: 14%; background-color: [cand-3-style-2];"><br></td>
            <td style="width: 14%; background-color: [cand-3-style-3];"><br></td>
        </tr>
    </tbody>
</table>
<hr>

"""

#############################################

exp_interface_1_1="""
<p><strong>The Mystery Author can be described as having the following styles:</strong></p>
<p><strong>For each style, we show how much it applies for each of the candidate authors</strong></p>
<p><strong>Style 1:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-1]</em></span></p>
<table style="width: 60%; background-color: rgb(209, 209, 209);border: none;">
    <tbody>
        <tr>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 1</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-1-style-1]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 2</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-2-style-1]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 3</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-3-style-1]</div>
            </td>
        </tr>
    </tbody>
</table>
<p><br><hr></p>


<p><strong>Style 2:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-2]</em></span></p>

<table style="width: 60%; background-color: rgb(209, 209, 209);border: none;">
    <tbody>
        <tr>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 1</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-1-style-2]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 2</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-2-style-2]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 3</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-3-style-2]</div>
            </td>
        </tr>
    </tbody>
</table>
<p><br><hr></p>

<p><strong>Style 3:</strong></p>
<p><span style="font-size: 16px; font-family: Helvetica;"><em>[style-3]</em></span></p>
<table style="width: 60%; background-color: rgb(209, 209, 209);border: none;">
    <tbody>
        <tr>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 1</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-1-style-3]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 2</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-2-style-3]</div>
            </td>
            <td style="width: 15%;">
                <div style="text-align: center;"><strong>Author 3</strong></div>
            </td>
            <td style="width: 5%;">
                <div style="text-align: left;">[cand-3-style-3]</div>
            </td>
        </tr>
    </tbody>
</table>
<p><br><hr></p>

"""

###########################################

exp_interface_2="""
<p><strong>The following presents a selection of writing style features our system identified in the texts and how much they apply for each of the authors</strong>(lighter color == high resemblance and darker color == low resemblance)</p>
<p><br></p>
<table style='"width:70%";'>
    <tbody>
        <tr>
            <td style="width: 30%; background-color: rgb(255, 255, 255);"><br></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Mystery Author</strong></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Author 1</strong></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Author 2</strong></td>
            <td style="width: 14%; background-color: rgb(209, 213, 216); text-align: center;"><strong>Author 3</strong></td>
        </tr>
        [table-body]
    </tbody>
</table>

"""