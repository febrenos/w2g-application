import React from "react";
import FormLogin from "../../../components/FormLogin";
import Planet from "../../../components/Planet";
import TitleApresentation from "../../../components/TitleApresentation";
import * as Styled from '../style'

export default function(){
    return(
        <Styled.Content>
            <Styled.Apresentation>
                <div>
                    <Planet width={200}/>
                    <TitleApresentation/>
                </div>
            </Styled.Apresentation>
            <FormLogin/>
        </Styled.Content>
    )
}