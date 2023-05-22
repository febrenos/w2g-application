import styled from "styled-components";

export const Content = styled.div`
width: 400px;
max-width: 480px;
background: rgb(32, 32, 36);
border-radius: 5px;
border-radius: 20px;
padding: 64px;
display: flex;
flex-direction: column;
justify-content: space-between;
display: grid;
grid-template-columns: repeat(1, 1fr);
grid-row-gap: 60px;
`

export const Img = styled.img`
    width:100%;
`

export const A = styled.a`
    text-decoration: none;
    color: #5271ff;
    font-weight: bold;
`